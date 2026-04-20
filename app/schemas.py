from pydantic import BaseModel

class PatientData(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Billing_Amount: float
    Admission_Type: str
    Insurance_Provider: str
    Medication: str
    Length_of_Stay: int