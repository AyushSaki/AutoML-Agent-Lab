import re

def analyse_requirements(topic, problem, users="", constraints="", success="", deployment="Flask"):
    text = " ".join([topic, problem]).lower()
    task_rules = [
        ("computer_vision", ["image", "photo", "visual defect", "object detection", "camera"]),
        ("nlp", ["text", "document", "chatbot", "language", "review", "summar"]),
        ("time_series", ["forecast", "forecasting", "future sales", "demand prediction", "time series"]),
        ("classification", ["classify", "classification", "fraud", "churn", "loan approval", "yes or no", "category"]),
        ("regression", ["price prediction", "predict price", "predict cost", "house price", "regression", "predict sales"]),
        ("clustering", ["segment", "segmentation", "group customers", "clustering"]),
        ("anomaly_detection", ["anomaly", "outlier detection", "detect defects", "unusual activity"]),
    ]

    detected = "needs_clarification"
    for task, keywords in task_rules:
        if any(keyword in text for keyword in keywords):
            detected = task
            break

    questions = []
    if not topic.strip():
        questions.append("What is the project topic and business problem?")
    if not problem.strip():
        questions.append("What decision or outcome should the system help with?")
    if not users.strip():
        questions.append("Who will use the system and what will they do with its output?")
    if not success.strip():
        questions.append("How will success be measured, and what level of error is acceptable?")
    if detected in ("regression", "classification") and not any(
        word in text for word in ["target", "column", "label", "price", "cost", "sales",
                                  "fraud", "churn", "approval", "predict"]
    ):
        questions.append("Which column should the model predict?")
    if not constraints.strip():
        questions.append("Are there constraints on budget, privacy, hardware, data volume, or tools?")

    phases = [
        {"phase": 1, "name": "Clarify requirements", "deliverable": "Approved project brief and success criteria"},
        {"phase": 2, "name": "Assess datasets", "deliverable": "Data quality, suitability, source and licence review"},
        {"phase": 3, "name": "Build baseline", "deliverable": "Preprocessing pipeline and baseline model"},
        {"phase": 4, "name": "Compare and validate", "deliverable": "Consistent holdout or cross-validation results"},
        {"phase": 5, "name": "Package delivery", "deliverable": "Model artifact, inference code, report and setup guide"},
    ]

    if detected == "needs_clarification":
        recommendation = "Confirm the problem type and intended output before selecting algorithms."
    elif detected == "regression":
        recommendation = "Predict a numeric value. Confirm the target column, units, and suitable error metrics."
    elif detected == "classification":
        recommendation = "Predict a class or category. Confirm the target column, class balance, and precision/recall priorities."
    elif detected == "clustering":
        recommendation = "Discover groups without a labelled target. Define how useful clusters will be judged."
    elif detected == "anomaly_detection":
        recommendation = "Identify unusual cases. Confirm whether labelled examples exist and how false alarms will be handled."
    elif detected == "nlp":
        recommendation = "Identify the language task, text fields, expected output, and any sensitive information."
    elif detected == "computer_vision":
        recommendation = "Confirm image types, defect/object labels, image volume, and annotation availability."
    else:
        recommendation = "Confirm the time column, forecast horizon, frequency, and baseline forecasting metric."

    return {
        "topic": topic.strip(),
        "business_problem": problem.strip(),
        "intended_users": users.strip(),
        "constraints": constraints.strip(),
        "success_criteria": success.strip(),
        "deployment": deployment.strip() or "To be decided",
        "detected_task": detected,
        "task_confidence": "Initial keyword-based suggestion; requires user confirmation",
        "recommendation": recommendation,
        "clarifying_questions": questions,
        "future_scope": phases,
        "data_policy": [
            "Validate actual dataset contents before making suitability claims.",
            "Record source and licence for externally obtained datasets.",
            "Clearly label synthetic data and never describe it as real-world data.",
            "Compare dataset quality separately from model performance.",
            "Use held-out data or cross-validation to reduce evaluation leakage.",
        ],
    }
