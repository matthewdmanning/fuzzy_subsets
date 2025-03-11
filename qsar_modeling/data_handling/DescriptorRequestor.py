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
        self.logger = None
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
        self.set_logger()

    def set_session_params(self):
        # if self._api_key and self._api_key: headers['api_key'] = API_KEY
        if self.user:
            self.default_headers["user-agent"] = self.user

    def set_logger(self):
        logging.basicConfig(
            filename="{}epa_data.log".format(os.environ.get("MODEL_DIR")),
            level=logging.INFO,
        )
        self.logger = logging.getLogger()
        self.logger.info("Logger initialized.")
        # sess.mount('https://', HTTPAdapter(max_retries=self.retries['total']))

    def bulk_epa_call(self, comlist):
        with requests.session() as r:
            # r.merge_environment_settings(url=self.api_url, proxies=None, verify=True, stream=False, cert=None)
            for mol_id in comlist:
                response, returned_input = self.make_api_call(
                    payload_input=mol_id, api_session=r
                )
                if not isinstance(response, dict):
                    if hasattr(response, "status_code"):
                        self.logger.warning(response.status_code)
                    else:
                        self.logger.warning(
                            "Response does not have a status code. {}".format(response)
                        )
                        self.logger.warning(response)
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
                    self.logger.warning(
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
                self.logger.warning("Connection error.", stack_level=3)
                attempts += 1
            except:
                self.logger.warning(
                    "{} error for {}".format(sys.exception(), payload_input)
                )
                attempts += 1
        if response is None:
            self.logger.warning("No response received for {}".format(payload_input))
            result = dict((self.input_type, payload_input))
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
        unpacked = dict()
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
        except ValueError:
            self.logger.warning(
                "Exception: {} for {}, {}".format(sys.exception(), api_input, response)
            )
            self.failed_responses[api_input] = sys.exception()
            unpacked = dict(("SMILES", api_input))
        return unpacked

    def parse_input(self, compounds):
        if isinstance(compounds, Sequence) and len(list(compounds)) > 0:
            data = list(compounds)
        elif isinstance(compounds, pd.DataFrame) or isinstance(compounds, pd.Series):
            data = compounds.tolist()
        elif isinstance(compounds, str):
            if "," in compounds:
                data = compounds.split(",")
            else:
                data = [compounds]
        else:
            self.logger.error("{}! That's not a SMILES string!".format(compounds))
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
        data = [
            pd.json_normalize(c[0])
            for c in self.bulk_epa_call(self.parse_input(compounds))
        ]
        df = pd.concat(data)
        self.logger.info("Successes: {}\n\n".format(len(self.response_dict.keys())))
        self.logger.info(
            "\n\nNumber of failed compounds: {}".format(
                len(self.failed_responses.keys())
            )
        )
        self.logger.info(df.head())
        return df, self.failed_responses


class QsarStdizer(ApiGrabber):
    _qsar_schema = {
        "filesInfo": [
            {
                "fileName": "string",
                "idField": "string",
                "strField": "string",
                "actField": "string",
            }
        ],
        "options": {"workflow": "string", "run": "string", "recordId": "string"},
    }

    # def __init__(self, api_url="https://hcd.rtpnc.epa.gov/api/stdizer/qsar-ready_08232023", payload=None):
    def __init__(
        self,
        api_url="https://hcd.rtpnc.epa.gov/api/stdizer",
        payload=None,
        *args,
        **kwargs
    ):
        # def __init__(self, api_url="https://ccte-cced-cheminformatics.epa.gov/api/stdizer", payload=None, *args, **kwargs):
        super().__init__(api_url=api_url, payload=payload, *args, **kwargs)
        if self.default_payload is None:
            self.default_payload = {"workflow": "qsar-ready"}  # _08232023"}


class ChemicalSearch(ApiGrabber):

    # def __init__(self, api_url="https://hcd.rtpnc.epa.gov/api/stdizer/qsar-ready_08232023", payload=None):
    def __init__(
        self,
        api_url="https://ccte-cced-cheminformatics.epa.gov/api/stdizer",
        payload=None,
        input_type="sid",
        *args,
        **kwargs
    ):

        # def __init__(self, api_url="https://ccte-cced-cheminformatics.epa.gov/api/stdizer", payload=None, *args, **kwargs):
        super().__init__(api_url=api_url, payload=payload, *args, **kwargs)
        if self.default_payload is None:
            self.default_payload = {"ids": [{"sim": 0}], "format": input_type}
        else:
            self.payload = payload


class DescriptorGrabber(ApiGrabber):
    SET_LIST = ["padel", "rdkit", "mordred", "toxprints"]
    # ?type=padel&smiles=ccff
    """
    {
      "options": {
        "workflow": "string",
        "run": "string",
        "recordId": "string"
      },
      "chemicals": [
        {
          "chemId": "string",
          "cid": "string",
          "sid": "string",
          "casrn": "string",
          "name": "string",
          "smiles": "string",
          "canonicalSmiles": "string",
          "inchi": "string",
          "inchiKey": "string",
          "mol": "string"
        }
      ],
      "full": true
    }
    """

    def __init__(
        self,
        desc_set,
        # api_url="https://hcd.rtpnc.epa.gov/api/descriptors",
        api_url="https://ccte-cced-cheminformatics.epa.gov/api/descriptors",
        *args,
        **kwargs
    ):
        super().__init__(api_url=api_url, payload={"type": desc_set}, *args, **kwargs)
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
