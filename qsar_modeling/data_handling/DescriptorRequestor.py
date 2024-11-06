import logging
import os
import sys
from collections.abc import Sequence

import pandas as pd
import requests
from urllib3 import Retry


class ApiGrabber:
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[502, 503, 504],
        allowed_methods={"POST", "GET"},
    )

    def __init__(
        self,
        api_url,
        headers=None,
        payload=None,
        user=None,
        timeout=(15, 300),
        input_type="smiles",
        flatten_output=True,
        url_method="GET",
        key_var=None,
    ):
        """

        :type api_url: str
        :type payload: dict
        :type headers: dict
        """
        # ?workflow = qsar - ready & smiles = CCC"
        self.default_headers = headers
        self.default_payload = payload
        self.api_url = api_url
        self.method = url_method
        self.input_type = input_type
        self.flatten_output = flatten_output
        self.response_dict = dict()
        self.info = None
        if key_var:
            self._api_key = os.environ[key_var]
        else:
            self._api_key = None
        self.user = user
        self.id_state = 0
        self.failed_responses = {}
        self.response_info = None
        self.timesouts = timeout
        self.set_session_params()

    def set_session_params(self):
        # if self._api_key and self._api_key: headers['api_key'] = API_KEY
        if self.user:
            self.default_headers["user-agent"] = self.user
        # sess.mount('https://', HTTPAdapter(max_retries=self.retries['total']))

    def bulk_epa_call(self, comlist):
        with requests.session() as r:
            # r.merge_environment_settings(url=self.api_url, proxies=None, verify=True, stream=False, cert=None)
            for mol_id in comlist:
                response, returned_input = self.make_api_call(
                    payload_input=mol_id, api_session=r
                )
                if isinstance(response, dict):
                    yield response, returned_input
                elif hasattr(response, "status_code"):
                    logging.warning(response.status_code)
                    yield response.status_code, returned_input
                else:
                    logging.warning(
                        "Response does not have a status code. {}".format(response)
                    )
                    logging.warning(response)
                    yield response, returned_input

    def make_api_call(self, payload_input, api_session):
        payload = self.default_payload.copy()
        self.id_state += 1
        payload[self.input_type] = payload_input
        attempts = 0
        response = None
        while attempts < self.retries.total:
            try:
                response = api_session.request(
                    method=self.method,
                    url=self.api_url,
                    params=payload,
                    timeout=self.timesouts,
                )
                if response.json():
                    break
                else:
                    logging.warning(
                        "Status code for response is {} for {}".format(
                            response.status_code, payload_input
                        )
                    )
                    attempts += 1
                """
                if response.status_code == 200:
                    break
                else:
                    print(response.status_code)
                """
            except (
                requests.exceptions.ConnectTimeout
                or requests.exceptions.RequestsWarning
                or requests.ConnectionError
            ):
                logging.warning("Connection error.")
                attempts += 1
            except:
                logging.warning(
                    "{} error for {}".format(sys.exception(), payload_input)
                )
                attempts += 1
        if response is None:
            logging.warning("No response received for {}".format(payload_input))
            result = dict(("smiles", payload_input))
        else:
            result = self.parse_api_response(response, payload_input)
        return result, payload_input

    def flatten(self, my_dict):
        result = dict()
        if isinstance(my_dict, list) and len(my_dict) == 1:
            return self.flatten(my_dict[0])
        elif isinstance(my_dict, dict):
            for key, value in my_dict.items():
                if isinstance(value, dict):
                    result.update(self.flatten(value))
                elif isinstance(value, list):
                    val_list = list()
                    for val in value:
                        if type(val) is dict:
                            result.update(self.flatten(val))
                        else:
                            val_list.append(value)
                    if len(val_list) > 0:
                        result[key] = value
                else:
                    result[key] = value
        else:
            print(my_dict)
            result = my_dict
        return result

    def parse_api_response(self, response, api_input):
        try:
            if type(response.json()) is list and len(response.json()) > 0:
                unpacked = self.flatten(response.json())
            elif type(response.json()) is dict and len(response.json().values()) > 0:
                unpacked = self.flatten(response.json())
        except AttributeError:
            if type(response) is list and len(response) > 0:
                unpacked = self.flatten(response)
            elif type(response) is dict and len(response.values()) > 0:
                unpacked = self.flatten(response)
        except:
            logging.warning(
                "Exception: {} for {}, {}".format(sys.exception(), api_input, response)
            )
            self.failed_responses[api_input] = sys.exception()
            unpacked = dict(("SMILES", api_input))
        return unpacked

    def parse_input(self, compounds):
        if isinstance(compounds, Sequence) and len(list(compounds)) > 0:
            data = compounds
        elif type(compounds) is str:
            if "," in compounds:
                data = compounds.split(",")
            else:
                data = [compounds]
        else:
            logging.error("{}! That's not a SMILES string!".format(compounds))
            raise TypeError
        return data

    def pandas_call(self, pd_input, column_name=None):
        if type(pd_input) is pd.DataFrame:
            return self.bulk_epa_call(pd_input[column_name].tolist())
        elif type(pd_input) is pd.Series:
            return self.bulk_epa_call(pd_input.tolist())
        else:
            raise IOError

    def grab_data(self, compounds):
        self.parse_input(compounds)
        df = pd.DataFrame(self.response_dict)
        logging.info("Successes: {}\n\n".format(len(self.response_dict.keys())))
        logging.info(
            "\n\nNumber of failed compounds: {}".format(
                len(self.failed_responses.keys())
            )
        )
        logging.info(df.head())
        return df, self.failed_responses


class QsarStdizer(ApiGrabber):

    def __init__(self, api_url="https://hcd.rtpnc.epa.gov/api/stdizer", payload=None):
        super().__init__(api_url=api_url, payload=payload)
        if self.default_payload is None:
            self.default_payload = {"workflow": "qsar-ready"}

    # ?type=padel&smiles=ccff


class DescriptorGrabber(ApiGrabber):
    SET_LIST = ["padel", "rdkit", "mordred", "toxprints"]

    def __init__(
        self,
        desc_set,
        api_url="https://hcd.rtpnc.epa.gov/api/descriptors",
        *args,
        **kwargs
    ):
        super().__init__(api_url=api_url, payload={"type": desc_set})
        self.desc_key_list = []

        if desc_set not in self.SET_LIST:
            print(
                "\n{}} was not in list of available descriptor sets.\n".format(desc_set)
            )
            raise TypeError

    def _parse_descriptor_response(self, response, api_input):
        if "descriptors" in response.json()["chemicals"][0]:
            self.response_dict[response.json()["chemicals"][0]["smiles"]] = (
                response.json()["chemicals"][0]["descriptors"]
            )
        else:
            logging.error(
                "Descriptors are missing in {} for {}\n.".format(
                    response.json(), api_input
                )
            )
            self.failed_responses[api_input] = response.json()
