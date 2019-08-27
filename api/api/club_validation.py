from marshmallow import Schema, fields
from marshmallow import ValidationError

import typing as t
import json


class InvalidInputError(Exception):
    """Invalid model input."""

class LendingClubSchema(Schema):
    addr_state = fields.Str()
    annual_inc = fields.Float()
    delinq_2yrs = fields.Float()
    dti = fields.Float()
    earliest_cr_line = fields.Str()
    emp_length = fields.Str()
    fico_average = fields.Float()
    grade = fields.Str()
    home_ownership = fields.Str()
    inq_last_6mths = fields.Float()
    installment = fields.Float()
    last_credit_pull_d = fields.Str()
    loan_amnt = fields.Float()
    open_acc = fields.Float()
    pub_rec = fields.Float()
    pub_rec_bankruptcies = fields.Float()
    purpose = fields.Str()
    revol_bal = fields.Float()
    revol_util = fields.Str()
    target = fields.Integer()
    term = fields.Str()
    title = fields.Str()
    total_acc = fields.Float()
    verification_status = fields.Str()

def _filter_error_rows(errors: dict,
                       validated_input: t.List[dict]
                       ) -> t.List[dict]:
    """Remove input data rows with errors."""
    
    indexes = errors.keys()
    # delete them in reverse order so that you
    # don't throw off the subsequent indexes.
    for index in sorted(indexes, reverse=True):
        #if isinstance(index, int): 
        validated_input = json.loads(validated_input)
        del validated_input[index]

    if len(validated_input) > 0:
        validated_input = json.dumps(validated_input)

        return validated_input

def validate_inputs(input_data):
    """Check prediction inputs against schema."""

    # set many=True to allow passing in a list
    schema = LendingClubSchema(many=True)

    errors = None

    try:
        schema.load(json.loads(input_data))
    except ValidationError as exc:
        errors = exc.messages
        
    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data
    
    return validated_input, errors