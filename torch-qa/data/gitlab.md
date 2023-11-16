Question: How to include other yml files in main .gitlab-ci.yml file?
Answer:

The content of main .gitlab-ci.yml file
```yml
include:
- local: '/configs/lua-anti-ddos.yml'
- local: '/configs/config.yml'

enable anti ddos lua:
  extends: .enable-lua-anti-ddos
  stage: YourStageName
  variables:
    INGRESS_NAME: YourIngressName
    NAMESPACE: YourNamespaceName
```

The content of /configs/lua-anti-ddos.yml file
```yml
.enable-lua-anti-ddos:
  when: manual
  interruptible: true
  script:
    - >
      kubectl patch ingress $INGRESS_NAME -n $NAMESPACE \
          --type=merge -p '''
              { "metadata":
                  { "annotations":
                    {
                        "nginx.ingress.kubernetes.io/configuration-snippet": "if ($uri ~ \"/$\"){access_by_lua_file /etc/nginx/lua/anti_ddos_challenge.lua;}"
                    }
                  }
             }''' \
         -o yaml
.disable-lua-anti-ddos:
  when: manual
  interruptible: true
  script:
    - >
      kubectl patch ingress $INGRESS_NAME -n $NAMESPACE \
        --type='json' \
        -p='[{"op": "remove", "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1configuration-snippet"}]' \
        -o yaml
```

---
Question: How to create a tag on a production project?
Answer:

We can use gitlab REST API to get the last SHA deployed to production and create a tag on it.
```yml
.Project-Create-Tag-Template: &Project-Create-Tag-Template
  when: manual
  image: badouralix/curl-jq:alpine
  script:
    - echo "Reading '$PROJECT_NAME''s last SHA deployed to 'Production'"
    - 'curl --request GET --header "PRIVATE-TOKEN: $GIT_TOKEN" --url "http://gitlab.com/api/v4/projects/$PROJECT_ID/environments/$ENVIRONMENT_ID" --silent > get-environment.json'
    - export ENVIRONMENT_SHA="$(jq --raw-output '.last_deployment.sha' get-environment.json)"
    - export TAGNAME="PROD-$(date +%Y%m%d%H%M%S)"
    - echo "Creating tag '$TAGNAME' on '$PROJECT_NAME' with SHA '$ENVIRONMENT_SHA'"
    - 'curl --request POST --header "PRIVATE-TOKEN: $GIT_TOKEN" --url "http://gitlab.com/api/v4/projects/$PROJECT_ID/repository/tags?tag_name=$TAGNAME&ref=$ENVIRONMENT_SHA" --silent'

Create-Tag-For-Project:
  extends: .Project-Create-Tag-Template
  stage: Trigger Project Release
  variables:
    PROJECT_NAME: YourProjectName
    PROJECT_ID: 701
    ENVIRONMENT_ID: 79
    GIT_TOKEN: glpat-XaaaX-11_xxxxxxTKxxxH    
```

---
Question: How to create file in ConfigMap?
Answer:

The content of xxx.yml file, If you wish create app-settings.json file,
```yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: YourName
  namespace: YourNamespaceName
data:
  app-settings.json: |-
    {
      "Key":
      {
        "Name": "xxx",
        "WhiteList": "Flash,Jack"
      }
    }
```

Then you can add following script block in .gitlab-ci.yml main file.
```yml
deploy:
  when: manual
  script:
    - cat xxx.yml | kubectl apply -f -
    - kubectl rollout restart YourProjectName -n YourNamespaceName
    - kubectl rollout status YourProjectName -n YourNamespaceName
```

---
Question: 如何建立 SSH-Key 存取 Gitlab ?
Answer:

請打開Git Bash命令提示符並輸入命令，如下所示
```bash
ssh-keygen
```
它會提示 `Enter file in which to save the key (/home/yourName/.ssh/id_rsa):`, 
只需鍵入文件名並按ENTER. 

* 登錄到您的GitLab帳戶，然後單擊[Settings] 選項
* 請單擊菜單左側的 `SSH Keys` 選項卡
* copy content of `/home/yourName/.ssh/id_rsa/id_rsa.pub` into Gitlab Key field
