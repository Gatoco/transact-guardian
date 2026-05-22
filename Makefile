.PHONY: help install train evaluate save-model inference clean

help:
	@echo "Comandos disponibles:"
	@echo "  make install      - Instalar dependencias"
	@echo "  make train        - Entrenar el modelo"
	@echo "  make evaluate     - Evaluar el modelo"
	@echo "  make save-model   - Guardar el modelo"
	@echo "  make inference    - Hacer predicciones"
	@echo "  make clean        - Limpiar archivos temporales"

install:
	pip install -r requirements.txt

train:
	python -c "from src.preprocess import preprocess_pipeline; \
	           from sklearn.linear_model import LogisticRegression; \
	           from sklearn.ensemble import RandomForestClassifier; \
	           X_train, y_train, X_test, y_test, _ = preprocess_pipeline('creditcard.csv'); \
	           model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1); \
	           model.fit(X_train, y_train); \
	           print('Modelo entrenado exitosamente')"

evaluate:
	python src/evaluate.py

save-model:
	python src/save_model.py

inference:
	python src/inference.py --input creditcard.csv --output predictions.csv

clean:
	rm -rf models/*.pkl
	rm -rf notebooks/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete