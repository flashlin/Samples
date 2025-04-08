plugins {
    kotlin("jvm") version "1.6.21"
    id("org.openjfx.javafxplugin") version "0.0.13"
    application
}

group = "com.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven {
        url = uri("https://oss.sonatype.org/content/repositories/snapshots")
    }
}

javafx {
    version = "15.0.1"
    modules = listOf("javafx.controls", "javafx.graphics", "javafx.fxml")
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
    implementation("org.jetbrains.kotlin:kotlin-reflect:1.6.21")
    implementation("no.tornado:tornadofx:1.7.20")
}

application {
    mainClass.set("com.example.MainKt")
}

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions {
        jvmTarget = "11"
        apiVersion = "1.6"
        languageVersion = "1.6"
    }
} 