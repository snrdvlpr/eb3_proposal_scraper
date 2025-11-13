import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from scripts.llm_pdf_extractor import extract_pdf_with_llm
from scripts.remote_llm import RemoteLLM

app = FastAPI(
    title="EB3 Proposal Rate Extractor API",
    description="Extract insurance plan rates from proposal PDFs using LLM",
    version="1.0.0",
)


class ExtractResponse(BaseModel):
    file: str
    carrier: Optional[str]
    plans: list
    status: str = "success"


@app.get("/")
async def root():
    return {
        "message": "EB3 Proposal Rate Extractor API",
        "endpoints": {
            "POST /extract": "Upload a PDF file to extract rate data",
            "POST /extract-batch": "Upload multiple PDF files to extract rate data",
            "POST /test-llm": "Test the LLM connection with a simple prompt",
            "GET /health": "Check API health status",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


class TestLLMRequest(BaseModel):
    system_prompt: Optional[str] = "You are a helpful assistant."
    user_prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0


class TestLLMResponse(BaseModel):
    response: str
    status: str = "success"


@app.post("/test-llm", response_model=TestLLMResponse)
async def test_llm(request: TestLLMRequest):
    """
    Simple endpoint to test the LLM connection.
    
    Send a text prompt and receive the LLM's response.
    Useful for verifying the LLM endpoint is accessible and working.
    """
    try:
        llm = RemoteLLM()
        response = await llm.chat(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return TestLLMResponse(response=response, status="success")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calling LLM: {str(e)}"
        )


async def process_pdf_upload(
    pdf: UploadFile,
    pages_per_chunk: int = 2,
    max_chars: int = 6000,
    save_output: bool = False,
) -> ExtractResponse:
    """
    Core function to process an uploaded PDF and extract rates.
    """
    if not pdf.filename or not pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Create temporary file to store uploaded PDF
    temp_dir = Path(tempfile.gettempdir()) / "eb3_extractor"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_pdf_path = None
    temp_output_path = None

    try:
        # Save uploaded file to temporary location (use UUID to avoid filename conflicts)
        file_id = str(uuid.uuid4())
        temp_pdf_path = temp_dir / f"upload_{file_id}.pdf"
        with open(temp_pdf_path, "wb") as f:
            content = await pdf.read()
            f.write(content)

        # Determine output path (optional, only if save_output is True)
        if save_output:
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_output_path = output_dir / f"{Path(pdf.filename).stem}.json"
        else:
            # Use a temporary output path that we'll read and delete
            temp_output_path = temp_dir / f"output_{file_id}.json"

        # Extract rates using LLM
        result = await extract_pdf_with_llm(
            pdf_path=temp_pdf_path,
            output_path=temp_output_path,
            pages_per_chunk=pages_per_chunk,
            max_chars=max_chars,
        )

        # Remove file path from result (not needed in API response)
        result.pop("file", None)

        return ExtractResponse(
            file=pdf.filename,
            carrier=result.get("carrier"),
            plans=result.get("plans", []),
            status="success",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    finally:
        # Cleanup temporary files (keep if save_output=True for output)
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()

        if not save_output and temp_output_path and temp_output_path.exists():
            temp_output_path.unlink()


@app.post("/extract", response_model=ExtractResponse)
async def extract_rates(
    pdf: UploadFile = File(..., description="PDF proposal file"),
    pages_per_chunk: int = 2,
    max_chars: int = 6000,
    save_output: bool = False,
):
    """
    Extract insurance plan rates from a PDF proposal.

    - **pdf**: The PDF file to process
    - **pages_per_chunk**: Number of pages to group per LLM prompt (default: 2)
    - **max_chars**: Maximum characters per chunk (default: 6000)
    - **save_output**: Whether to save output JSON to disk (default: False)

    Returns extracted carrier name, plan names/IDs, rate structures, and rates.
    """
    return await process_pdf_upload(pdf, pages_per_chunk, max_chars, save_output)


@app.post("/extract-batch")
async def extract_batch(
    pdfs: list[UploadFile] = File(..., description="Multiple PDF proposal files"),
    pages_per_chunk: int = 2,
    max_chars: int = 6000,
    save_output: bool = False,
):
    """
    Extract rates from multiple PDF files in batch.

    Returns a list of extraction results, one per PDF.
    """
    results = []
    errors = []

    for pdf in pdfs:
        try:
            result = await process_pdf_upload(
                pdf=pdf,
                pages_per_chunk=pages_per_chunk,
                max_chars=max_chars,
                save_output=save_output,
            )
            results.append(result.model_dump())
        except HTTPException as e:
            errors.append({"file": pdf.filename or "unknown", "error": e.detail})
        except Exception as e:
            errors.append({"file": pdf.filename or "unknown", "error": str(e)})

    return {
        "results": results,
        "errors": errors,
        "total": len(pdfs),
        "successful": len(results),
        "failed": len(errors),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

