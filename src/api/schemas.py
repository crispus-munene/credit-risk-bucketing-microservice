from pydantic import BaseModel
from typing import Optional

from pydantic import BaseModel

class UserInput(BaseModel):
    loan_amnt: float
    funded_amnt: float
    term: int
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    home_ownership: str
    verification_status: str
    purpose: str
    addr_state: str
    dti: float
    open_acc: float
    revol_util: float
    initial_list_status: str
    last_pymnt_amnt: float
    application_type: str
    acc_open_past_24mths: float
    avg_cur_bal: float
    bc_open_to_buy: float
    bc_util: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mort_acc: float
    num_actv_rev_tl: float
    log_annual_inc: float
    fico_score: float
    credit_age_months: int
    credit_age_years: float

class PredictionResponse(BaseModel):
    cluster: int
    predicted_risk: str
    probability: float

