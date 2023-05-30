#podman run -d -p 6080:6080 -e EMULATOR_DEVICE="Samsung Galaxy S6" -e WEB_VNC=true --device /dev/kvm --name android-container budtmo/docker-android:emulator_11.0
podman stop android-container 
podman rm android-container
#podman run -d -p 6080:6080 -e DEVICE="Samsung Galaxy S6" -e WEB_VNC=true --name android-container budtmo/docker-android:emulator_10.0
podman run -d -p 6080:6080 -e DEVICE="Nexus 5" -e WEB_VNC=true --name android-container butomo1989/docker-android-x86-7.1.1
