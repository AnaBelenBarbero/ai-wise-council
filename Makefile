.PHONY: run-train run-docker upload-container-to-gcr

run-train:
	poetry run python ai_wise_council/train.py

run-docker:
	docker build -t ai-wise-council .
	docker run ai-wise-council

upload-container-to-gcr:
	for /f "delims=" %%i in ('gcloud config list project --format "value(core.project)"') do set PROJECT_ID=%%i
	set REPO_NAME=vertex-repository
	set IMAGE_NAME=ai-wise-council
	set IMAGE_TAG=v0.0.1
	set IMAGE_URI=us-central1-docker.pkg.dev/!PROJECT_ID!/!REPO_NAME!/!IMAGE_NAME!:!IMAGE_TAG!
	docker build -f Dockerfile -t !IMAGE_URI! ./
	docker push !IMAGE_URI!

upd:
	docker build -f Dockerfile -t us-central1-docker.pkg.dev/prefab-passage-446711-h8/vertex-repository/ai-wise-council:v0.0.1 ./
	docker push us-central1-docker.pkg.dev/prefab-passage-446711-h8/vertex-repository/ai-wise-council:v0.0.1