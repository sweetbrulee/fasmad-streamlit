# Getting Started
## Preparation
> NOTE: Python == 3.9.x is required
### Install with pip
```pip install -r requirements.txt```

### Setup Twilio token and SID
1. Sign up for a free account on [Twilio](https://twilio.com/)
2. Copy the auth token and account SID from the [Twilio console](https://www.twilio.com/console)
3. Add the auth token and account SID to the environment variables
    - ```export TWILIO_ACCOUNT_SID=<your account SID>```
    - ```export TWILIO_AUTH_TOKEN=<your auth token>```
    
## Usage
### Run the script
```. script/start.sh```

### Open the app
```https://localhost:8000``` 
    
    NOTE: prompt about insecure connection is normal in development
