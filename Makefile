.PHONY: run-train run-docker

run-train:
	poetry run python ai_wise_council/train.py

run-docker:
	docker build -t ai-wise-council .
	docker run ai-wise-council
