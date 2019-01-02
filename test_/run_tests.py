from test_._test_saving_loading import run_test_save_load
from test_._test_overfitting_dummy_model import _test_overfit_dummy
from test_._test_scores import    _1test_exact_match, _1test_edit_inverse

print("Starting to test ")
_test_overfit_dummy()
_1test_exact_match()
_1test_edit_inverse()
run_test_save_load()
print("**** Test all passed ****")