#!/bin/bash

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "Environment variables loaded successfully."
else
    echo "Warning: .env file not found. Using default environment variables."
fi

# Check if NUGET_API_KEY is set
if [ -z "$NUGET_API_KEY" ]; then
    echo "Error: NUGET_API_KEY is not set. Please set NUGET_API_KEY in .env file."
    exit 1
fi

# Array of project names
projectNames=(
    "T1.Standard"
    "T1.SqlSharp"
    "T1.SlackSDK"
    "T1.GrpcProtoGenerator"
)

# Show project selection with fzf
echo "Please select project to deploy to nuget:"
selectedProjectName=$(printf '%s\n' "${projectNames[@]}" | fzf --height=40% --reverse --border)

# Check if a project was selected
if [ -z "$selectedProjectName" ]; then
    echo "No project selected. Exiting..."
    exit 1
fi

echo "Selected project: $selectedProjectName"

# Change to the selected project directory
if [ ! -d "../$selectedProjectName" ]; then
    echo "Error: Directory ../$selectedProjectName does not exist."
    exit 1
fi

cd "../$selectedProjectName/$selectedProjectName"
echo "Changed to directory: $(pwd)"

# Function to publish to nuget
publish_to_nuget() {
    local project_name=$1
    
    echo "Publishing $project_name to nuget..."
    
    # Check if .csproj file exists
    if [ ! -f "$project_name.csproj" ]; then
        echo "Error: $project_name.csproj file not found in current directory."
        return 1
    fi
    
    # Build the project
    echo "Building project..."
    dotnet build --configuration Release
    
    if [ $? -ne 0 ]; then
        echo "Error: Build failed."
        return 1
    fi
    
    # Pack the project
    echo "Packing project..."
    rm -rf ./nupkg
    dotnet pack --configuration Release --output ./nupkg
    
    if [ $? -ne 0 ]; then
        echo "Error: Pack failed."
        return 1
    fi
    
    # Find the nupkg file
    nupkg_file=$(find ./nupkg -name "*.nupkg" | head -1)
    
    if [ -z "$nupkg_file" ]; then
        echo "Error: No .nupkg file found."
        return 1
    fi
    
    echo "Found nupkg file: $nupkg_file"
    
    # Publish to nuget
    echo "Publishing to nuget..."
    dotnet nuget push "$nupkg_file" --source ${NUGET_SOURCE:-https://api.nuget.org/v3/index.json} --api-key "$NUGET_API_KEY"
    
    if [ $? -eq 0 ]; then
        echo "Successfully published $project_name to nuget!"
    else
        echo "Error: Failed to publish to nuget."
        return 1
    fi
}

# Call the publish function
publish_to_nuget "$selectedProjectName"
