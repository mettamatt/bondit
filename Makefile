.PHONY: format lint typecheck all

SEPARATOR = "--------------------------------------------------"

format:
	@echo $(SEPARATOR)
	@echo "🔧 Running black..."
	@echo $(SEPARATOR)
	@python -m black .
	@echo $(SEPARATOR)
	@echo "✅ Black done!"
	@echo $(SEPARATOR)
	@echo ""

	@echo $(SEPARATOR)
	@echo "🔧 Running isort..."
	@echo $(SEPARATOR)
	@python -m isort .
	@echo $(SEPARATOR)
	@echo "✅ Isort done!"
	@echo $(SEPARATOR)
	@echo ""

lint:
	@echo $(SEPARATOR)
	@echo "🔍 Running flake8..."
	@echo $(SEPARATOR)
	@python -m flake8 .
	@echo $(SEPARATOR)
	@echo "✅ Flake8 done!"
	@echo $(SEPARATOR)
	@echo ""

typecheck:
	@echo $(SEPARATOR)
	@echo "🔍 Running mypy..."
	@echo $(SEPARATOR)
	@python -m mypy .
	@echo $(SEPARATOR)
	@echo "✅ Mypy done!"
	@echo $(SEPARATOR)
	@echo ""

all: format lint typecheck
	@echo $(SEPARATOR)
	@echo "🎉 All checks are complete! 🎉"
	@echo $(SEPARATOR)
