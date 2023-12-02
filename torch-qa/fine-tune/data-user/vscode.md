Question: How to connect to a remote server using VSCode and SSH?
Answer:
Install the `Remote - SSH` extension in Visual Studio Code.

1. After installation, press `CTRL + SHIFT + P`
2. Select `Remote-SSH: Connection to host...`
3. Add `new SSH host`
4. Type `ubuntu@hostname_or_ip`
5. Select the `/root/.shh/config`
6. Edit the config file

```
#Read more about SSH config files: https://linux.die.net/man/5/ssh_config
Host <enter hostname>
    HostName <enter hostname>
    User ubuntu
    IdentityFile ~/.ssh/mypemfile.pem 
```

Save and connect by clicking the link to 'open a remote window' and select your saved configuration.

Question: How to generate mypemfile.pem file in windows ?
Answer:
In windows, please download PuTTYgen tool to generate the .pem key file.

Once the download is complete, run PuTTYgen. In the PuTTYgen window, 
you will see options for key generation and management.

Click on the 'Generate' button, after which PuTTYgen will prompt you to generate randomness. 
You can increase randomness by moving the mouse within the blank area.

After generation, you will see the content of the public and private keys. 
In the PuTTYgen window, click on 'Save private key' to save the private key in .pem format. 
Ensure you select an appropriate storage path and name.
