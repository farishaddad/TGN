.PHONY: setup app learn test lint clean

# Setup the project
setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

# Launch the Streamlit app
app:
	. .venv/bin/activate && streamlit run app/main.py

# Run all learning scripts in sequence
learn:
	. .venv/bin/activate && python learn/01_data_generation.py
	. .venv/bin/activate && python learn/02_graph_construction.py
	. .venv/bin/activate && python learn/03_tgn_architecture.py
	. .venv/bin/activate && python learn/04_training_loop.py
	. .venv/bin/activate && python learn/05_inference_scoring.py
	. .venv/bin/activate && python learn/06_mint_transfer.py

# Run all tests
test:
	. .venv/bin/activate && python -m pytest tests/ -v --tb=short

# Run linter
lint:
	. .venv/bin/activate && ruff check tgn_learn/ tests/ learn/

# Quick training demo
train:
	. .venv/bin/activate && python -m tgn_learn.training --epochs 10

# Generate sample data
generate:
	. .venv/bin/activate && python -m tgn_learn.generators --type banksim --accounts 200 --transactions 5000

# Clean up generated files
clean:
	rm -rf checkpoints/ .pytest_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
