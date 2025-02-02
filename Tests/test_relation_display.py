import pytest
from relations import int2str, int2matrix

def test_int2str_9bit():
    assert int2str(0b101010101, length=9) == '101010101'
    assert int2str(1, length=9) == '000000001'
    assert int2str(0b111111111, length=9) == '111111111'

def test_int2str_27bit():
    assert int2str(0b101010101010101010101010101, length=27) == '101010101010101010101010101'
    assert int2str(0b01, length=27) == '000000000000000000000000001'
    assert int2str(2, length=27) == '000000000000000000000000010'
    assert int2str(0b111111111111111111111111111, length=27) == '111111111111111111111111111'

def test_int2str_value_error():
    with pytest.raises(ValueError):
        int2str(0b1111111111111111111111111111, length=27)

def test_int2matrix():
    relation_int = 0b100010001010101111101010000
    expected_output = (
        '|100|\t|010|\t|101|\n'
        '|010|\t|101|\t|010|\n'
        '|001|\t|111|\t|000|\n'
    )
    assert int2matrix(relation_int) == expected_output

def test_int2matrix_with_indent():
    relation_int = 0b101010101010101010101010101
    expected_output = (
        '    |101|\t|010|\t|101|\n'
        '    |010|\t|101|\t|010|\n'
        '    |101|\t|010|\t|101|\n'
    )
    assert int2matrix(relation_int, indent='    ') == expected_output
