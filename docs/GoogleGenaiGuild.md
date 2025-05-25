# Google GenAI SDK Function Calling Example

Show how to register, call, and handle a function with the `google-genai` SDK.

```python
import os
from google import genai
from google.genai import types

# 1. Configure client with API key
os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'
client = genai.Client()

# 2. Define function metadata
weather_fn = {
    'name': 'get_current_temperature',
    'description': 'Get current temperature for a location',
    'parameters': {
        'type': 'object',
        'properties': {
            'location': {'type': 'string', 'description': 'City name, e.g. Taipei'}
        },
        'required': ['location']
    }
}

# 3. Send request with function declaration
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=['What is the temperature in Taipei?'],
    functions=[weather_fn],
    function_call={'name': 'get_current_temperature'}
)

# 4. Invoke the function if called
call = response.candidates[0].content.parts[0].function_call
if call:
    args = call.arguments
    temp = get_current_temperature(**args)  # your implementation
    print(f"Temperature: {temp['value']} {temp['unit']}")
else:
    print(response.text)
```
