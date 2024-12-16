# OpenVINO™ backend on MiniCPM3-4B
* Model: https://huggingface.co/openbmb/MiniCPM3-4B
* OS: Windows 11
* CPU: Intel(R) Core(TM) Ultra 7 258V
* Note: Using OpenVINO pre-release version. [`URL`](https://storage.openvinotoolkit.org/wheels/pre-release/)
``` sh
openvino==2024.6.0rc3
openvino-genai==2024.6.0.0rc3
openvino-telemetry==2024.5.0
openvino-tokenizers==2024.6.0.0rc3
```

# Step 1. create python environment

``` sh
conda create -n ov_minicpm_4B python=3.10
conda activate ov_minicpm_4B
pip install -r requirements.txt
```

# Step2. Convert MiniCPM3-4B to OpenVINO™ IR. Be patient, it may takes some time.
``` sh
python convert_minincmp3-4B.py -m /path/to/minicpm3-4b -o /path/to/minicpm3-4b_ov
```
<img src="./images/1.png" width="50%"></img>
# Step3. Testing
``` sh
python chat_minicpm3-4B.py -m /path/to/inicpm3-4b_ov -d ( "CPU" or "GPU")
```
<img src="./images/2.png" width="50%"></img>
