rom fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import List
from predict import predict_data
import joblib

app = FastAPI(
    title="Iris Classification API",
    description="API for predicting Iris flower species using Random Forest",
    version="2.0.0"
)

# Load model once at startup for better performance
model = joblib.load("../model/iris_model.pkl")

# Data Models
class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class IrisResponse(BaseModel):
    response: int
    
class IrisPredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float

class IrisBatchData(BaseModel):
    samples: List[IrisData]

class IrisBatchResponse(BaseModel):
    predictions: List[int]
    class_names: List[str]
    count: int

class IrisProbabilityResponse(BaseModel):
    predicted_class: int
    class_name: str
    probabilities: dict
    confidence: float


# Helper function to get class name
def get_class_name(class_id: int) -> str:
    """Convert class ID to Iris species name"""
    class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return class_names.get(class_id, "unknown")


# Endpoint 1: Health Check
@app.get("/", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_ping():
    """
    Health check endpoint to verify API is running
    """
    return {
        "status": "healthy",
        "message": "Iris Classification API is running",
        "model": "RandomForestClassifier"
    }


# Endpoint 2: Basic Prediction (Your original endpoint - kept as is)
@app.post("/predict", response_model=IrisResponse, tags=["Prediction"])
async def predict_iris(iris_features: IrisData):
    """
    Predict Iris species class (returns only class number)
    
    - **sepal_length**: Length of sepal in cm
    - **sepal_width**: Width of sepal in cm
    - **petal_length**: Length of petal in cm
    - **petal_width**: Width of petal in cm
    
    Returns: Class ID (0: setosa, 1: versicolor, 2: virginica)
    """
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]
        prediction = predict_data(features)
        return IrisResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 3: Prediction with Class Name and Confidence
@app.post("/predict/detailed", response_model=IrisPredictionResponse, tags=["Prediction"])
async def predict_iris_detailed(iris_features: IrisData):
    """
    Predict Iris species with class name and confidence score
    
    Returns detailed prediction including:
    - Class ID
    - Class name (species)
    - Confidence score
    """
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]
        
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        
        return IrisPredictionResponse(
            predicted_class=int(prediction[0]),
            class_name=get_class_name(int(prediction[0])),
            confidence=round(confidence * 100, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 4: Prediction with All Probabilities
@app.post("/predict/probabilities", response_model=IrisProbabilityResponse, tags=["Prediction"])
async def predict_with_probabilities(iris_features: IrisData):
    """
    Predict Iris species with probability scores for all classes
    
    Returns:
    - Predicted class ID and name
    - Probability distribution across all three species
    - Overall confidence
    """
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]
        
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]
        
        class_names = ["setosa", "versicolor", "virginica"]
        prob_dict = {
            class_names[i]: round(float(probabilities[i]) * 100, 2) 
            for i in range(len(probabilities))
        }
        
        return IrisProbabilityResponse(
            predicted_class=int(prediction[0]),
            class_name=get_class_name(int(prediction[0])),
            probabilities=prob_dict,
            confidence=round(float(max(probabilities)) * 100, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 5: Batch Prediction
@app.post("/predict/batch", response_model=IrisBatchResponse, tags=["Prediction"])
async def predict_batch(data: IrisBatchData):
    """
    Predict multiple Iris samples at once
    
    Useful for batch processing of multiple flowers
    """
    try:
        if len(data.samples) == 0:
            raise HTTPException(status_code=400, detail="No samples provided")
        
        predictions = []
        class_names_list = []
        
        for sample in data.samples:
            features = [[sample.sepal_length, sample.sepal_width,
                        sample.petal_length, sample.petal_width]]
            prediction = model.predict(features)
            predictions.append(int(prediction[0]))
            class_names_list.append(get_class_name(int(prediction[0])))
        
        return IrisBatchResponse(
            predictions=predictions,
            class_names=class_names_list,
            count=len(predictions)
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 6: Model Information
@app.get("/model/info", tags=["Model Info"])
async def model_info():
    """
    Get information about the trained Random Forest model
    
    Returns model metadata and configuration
    """
    try:
        return {
            "model_type": "RandomForestClassifier",
            "n_estimators": int(model.n_estimators),
            "max_depth": int(model.max_depth) if model.max_depth else None,
            "n_features": int(model.n_features_in_),
            "n_classes": int(model.n_classes_),
            "classes": model.classes_.tolist(),
            "class_names": ["setosa", "versicolor", "virginica"],
            "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 7: Feature Importance
@app.get("/model/feature-importance", tags=["Model Info"])
async def feature_importance():
    """
    Get feature importance scores from the Random Forest model
    
    Shows which features are most important for classification
    """
    try:
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        importances = model.feature_importances_
        
        feature_imp_dict = {
            feature_names[i]: round(float(importances[i]) * 100, 2)
            for i in range(len(feature_names))
        }
        
        # Sort by importance
        sorted_features = sorted(feature_imp_dict.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "feature_importances": feature_imp_dict,
            "ranked_features": [{"feature": f[0], "importance": f[1]} for f in sorted_features],
            "most_important_feature": sorted_features[0][0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 8: Model Statistics
@app.get("/model/statistics", tags=["Model Info"])
async def model_statistics():
    """
    Get statistical information about the model's trees
    
    Provides insights into the Random Forest ensemble
    """
    try:
        n_trees = len(model.estimators_)
        tree_depths = [tree.get_depth() for tree in model.estimators_]
        tree_nodes = [tree.tree_.node_count for tree in model.estimators_]
        
        return {
            "number_of_trees": n_trees,
            "average_tree_depth": round(sum(tree_depths) / len(tree_depths), 2),
            "max_tree_depth": max(tree_depths),
            "min_tree_depth": min(tree_depths),
            "average_nodes_per_tree": round(sum(tree_nodes) / len(tree_nodes), 2),
            "total_nodes": sum(tree_nodes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 9: API Information
@app.get("/info", tags=["Info"])
async def api_info():
    """
    Get general API information and usage guidelines
    """
    return {
        "api_name": "Iris Classification API",
        "version": "2.0.0",
        "description": "ML API for classifying Iris flowers using Random Forest",
        "model": "RandomForestClassifier",
        "endpoints": {
            "health": "/",
            "basic_prediction": "/predict",
            "detailed_prediction": "/predict/detailed",
            "probability_prediction": "/predict/probabilities",
            "batch_prediction": "/predict/batch",
            "model_info": "/model/info",
            "feature_importance": "/model/feature-importance",
            "model_statistics": "/model/statistics"
        },
        "supported_species": ["setosa", "versicolor", "virginica"],
        "features_required": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    }
    
