import uvicorn
from app.main import app

try:
    import spaces
except ImportError:
    class DummySpaces:
        def GPU(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    spaces = DummySpaces()

@spaces.GPU()
def dummy_function():
    pass

if __name__ == "__main__":
    # Hugging Face Spaces exposes port 7860
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)
