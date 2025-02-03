.PHONY: run-train run-docker upload-container-to-gcr

run-train:
	poetry run python ai_wise_council/train.py

run-docker-dev:
	docker build -f docker-dev/Dockerfile -t ai-wise-council-dev .
	docker run ai-wise-council

docker-build-train:
	docker build -f Dockerfile -t anabelenbarbero/ai-wise-council:v0.0.5 ./
	
push-to-docker-hub-train:
	docker push anabelenbarbero/ai-wise-council:v0.0.5

docker-build-predict:
	docker build -f Dockerfile_predict -t anabelenbarbero/ai-wise-council:v0.0.1_predict ./

push-to-docker-hub-predict:
	docker push anabelenbarbero/ai-wise-council:v0.0.1_predict

run-api-dev:
	poetry install
	poetry run fastapi dev ai_wise_council/predict.py 



