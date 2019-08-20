import math
from lending_club.predict import make_prediction, load_dataset, load_pipeline

def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='lending_club_selected_features_test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert subject.get('predictions')[0] == 1


def test_make_multiple_predictions():
    # Given
    test_data = load_dataset(file_name='lending_club_selected_features_test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient='records')

    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 7848
