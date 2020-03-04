"""Implements utilities for parsing and writing stings to be used to
communicate between DESDEO and ComVis.

"""
from typing import List, Union, Tuple, Dict, Any


OK_TOKEN = "OK"
ERROR_TOKEN = "ERR"
END_TOKEN = "END"
DATA_TOKEN = "DATA"
MSG_TOKEN = "MSG"
DELIMITER = "_"

def strip_start_and_end(message: str,
                     start_tokens: List[str] = [OK_TOKEN, ERROR_TOKEN],
                     end_token: str = END_TOKEN,
                     delimiter: str = DELIMITER,
) -> Union[Tuple[str, str], bool]:
    # strips the start and end tokens of a message, returning the stripped message and the
    # start token. If message is invalid, return False.

    # check end token
    split = message.split(delimiter)
    if not split[-1] == end_token:
        return False

    # check start token
    found_start = None
    for token in start_tokens:
        if split[0] == token:
            found_start = token
            break

    if found_start is None:
        # no valid start token found
        return False

    stripped_msg = delimiter.join(split[1:-1])
    return  (stripped_msg, found_start)


def to_dict(message: str,
            tokens: List[str] = [DATA_TOKEN, MSG_TOKEN],
            delimiter: str = DELIMITER,
) -> Dict[str, str]:
    # Parses a message into a dictionary with the keys being the tokends, and the value whatever follows
    # the delimiter after the token. Ignores anything that is not specified in toekns.
    # Assumes each token to be present only once, if multiples, only the first occurrence of the
    # tooken will be accounted for

    split = message.split(delimiter)
    res_d = {}

    for token in tokens:
        try:
            i = split.index(token)

            # token found
            res_d[token] = split[i+1]
        except ValueError:
            # token not found
            pass

    return res_d


def parse_message(message: str,
                  start_tokens: List[str] = [OK_TOKEN, ERROR_TOKEN],
                  end_token: str = END_TOKEN,
                  payload_tokens: List[str] = [DATA_TOKEN, MSG_TOKEN],
                  delimiter:str = DELIMITER
) -> Dict[str, str]:
    # Parses a  whole message into a dictionary with keys from  payload_tokens.
    # The payload is whatever comes after the delimiter, and is assumed to end when the next
    # delimeter is reached. The dist will have an extra key STATE to indicate wat kind of strart the message had.
    # Return the dict, if errors are encountered, the dict will have only an ERR key, with a string explaing
    # the cause og the error.
    stripped_msg = strip_start_and_end(message,
                                       start_tokens=start_tokens,
                                       end_token=end_token,
                                       delimiter=delimiter)



    if not stripped_msg:
        res_d = {"ERROR": "Invalid start or end of message."}
        return res_d

    payload_d = to_dict(stripped_msg[0])

    if not payload_d:
        res_d = {"ERROR": "No payload tokens found."}
        return res_d

    payload_d["STATE"] =  stripped_msg[1]
    return payload_d


def parse_dict(dictionary: Dict[str, Any],
               end_token: str = END_TOKEN,
               delimiter: str = DELIMITER
) -> str:
    # TAkes a dictionary, parses it into a string with tokens being the same as the keys of the dict, followed
    # by the associated value as a string. The string will start with whatever value STATE is associated with,
    # and end with the end_token. The delimeter in the string is the delimiter.
    # In case of errors, returns an empty string.
    if "STATE" not in dictionary:
        print(f"No STATE key found in {dictionary}")
        return ""

    payload = [f"{key + delimiter + str(dictionary[key])}" for key in dictionary if key != "STATE"]
    string = delimiter.join([dictionary["STATE"]] + payload + [f"{END_TOKEN}"])

    return string


if __name__=="__main__":
    test_str = "OK_DATA_[[1,2,3], [4,5,6]]_MSG_[all is well]_END"
    test_str2= "ERR_MSG_[error message]_END"
    test_start = "START_something_something_END"
    test_end = "OK_something_something_TERMINATE"
    test_payload = "OK__END"

    res = parse_message(test_str)
    res2 = parse_message(test_str2)
    res_start = parse_message(test_start)
    res_end = parse_message(test_end)
    res_payload = parse_message(test_payload)
    
    str_str = parse_dict(res)
    str_str2 = parse_dict(res2)

    print("Example 1:")
    print(f"Input: {test_str}\nOutput: {res}")
    print("Example 2:")
    print(f"Input: {test_str2}\nOutput: {res2}")

    assert test_str == str_str
    assert test_str2 == str_str2
    print("Test ok")

    test_example = "OK_DATA_[[1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3]]_END"
    print(parse_dict(parse_message(test_example)))
