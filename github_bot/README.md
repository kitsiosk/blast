This is the source code for the bot that can be deployed in a GitHub repository and whenever a PR that is linked to an issue is opened, it will automatically run BLAST and try to generate a fail-to-pass test for that PR-issue pair. If such test is generated, the bot will post it as a comment to the PR discussion.

# Setup
Python3.12 is needed. Inside a virtual environment, install the required packages by running:
```sh
pip install -r requirements.txt
```

Then, you need to setup your LLM API keys:
```sh
export OPENAI_API_KEY="sk-XXXX"   # required
export GROQ_API_KEY="gsk-YYYY"    # optional (for ablation studies with LLaMA/DeepSeek)
```

Your GitHub PAT is also needed to post the comment:
```
export GITHUB_PAT="ghp_XXX"
```


# Testing the bot locally
You can test the bot locally by mocking the opening of a PR using the PR's payload, like in `webhook_handler/test_mocks/bugbug_4867.json`, which is a mock payload for [this](https://github.com/mozilla/bugbug/pull/4867) PR. Then, you can test the bot locally by running

```sh
python manage.py test webhook_handler.tests.TestGenerationBugbug4867
```

# Deploying the bot
To deploy the bot to your repository, you will need
1. Full access to the repository,
2. A machine to host the bot. We have tested it in an Ubuntu 22.06 machine. The bot is generally lightweight so a basic setup should be enough.

### Spin up the Django server

These are the steps to spin up and deploy the Django server with **Gunicorn** and **Nginx** on Ubuntu.

###### 1. Install Nginx
```sh
sudo apt install nginx -y
```

###### 2. Apply Database Migrations
```sh
python manage.py migrate
```

###### 3. Test Gunicorn
Run Gunicorn manually to verify everything works:

```sh
gunicorn --bind 0.0.0.0:8000 github_bot.wsgi
```

Hit **Ctrl+C** after confirming the server responds.


###### 4. Create a Systemd Service
Create and edit the service file:

```sh
sudo nano /etc/systemd/system/django_webhook.service
```

Paste the following configuration (adjust paths if necessary):

```ini
[Unit]
Description=Django Webhook Server
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/gh-bot
ExecStart=/home/ubuntu/gh-bot/.venv/bin/gunicorn --workers 3 --bind 0.0.0.0:8000 github_bot.wsgi
Restart=always

[Install]
WantedBy=multi-user.target
```

Reload systemd and enable the service:

```sh
sudo systemctl daemon-reload
sudo systemctl enable django_webhook
sudo systemctl start django_webhook
sudo systemctl status django_webhook
```

###### 5. Configure Nginx
Create a new Nginx site configuration:

```sh
sudo nano /etc/nginx/sites-available/django_webhook
```

Insert the following (replace `130.60.24.144` with your server’s IP or domain):

```nginx
server {
   listen 80;
   server_name 130.60.24.144;

   location / {
       proxy_pass http://127.0.0.1:8000;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
       proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
   }
}
```

Enable the configuration and restart Nginx:

```sh
sudo ln -s /etc/nginx/sites-available/django_webhook /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```


###### 6. Configure Firewall
Allow SSH and HTTP traffic:

```sh
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw enable
```


###### 7. Verify Deployment
Open your browser and visit:

```
http://130.60.24.144
```

You should see your Django webhook running successfully!


### Configure your GitHub repo to trigger the bot on new PRs
- Go your GitHub repo → Settings → Webhooks → Add webhook.
- Set the Payload URL to `http://130.60.24.144/webhook/`
- Set Content Type to `application/json`
- Fill in your secret, which has to match the variable `GITHUB_WEBHOOK_SECRET = "1234"` in `webhook_handler/views.py`.
- Select `Let me select individual events.` and then check the `Pull Requests` box.
- Click `Add webhook`.
- Test that the webhook is working by checking the logs: `sudo journalctl -u django_webhook --follow`
    - If you open a PR in your repo, you should see the associated logs.