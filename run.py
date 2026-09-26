import uvicorn
from app.main import app

try:
    import spaces
except ImportError:
    class DummySpaces:
        def GPU(self, func=None, *args, **kwargs):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func
    spaces = DummySpaces()

@spaces.GPU
def dummy_function():
    pass

if __name__ == "__main__":
    # Hugging Face Spaces exposes port 7860
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)
