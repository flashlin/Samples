#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

char** getDrives() {
    int drives = GetLogicalDrives();
    int count = 0;
    char** result = (char**)malloc(sizeof(char*) * 26);
    int i = 0;
    for (i = 0; i < 26; i++) {
        if ((drives & (1 << i)) != 0) {
            char* driveLetter = (char*)malloc(sizeof(char) * 2);
            driveLetter[0] = (char)(i + 'A');
            driveLetter[1] = '\0';
            result[count] = driveLetter;
            count++;
        }
    }
    result[count] = NULL;
    return result;
}

void freeStringArray(char** stringArray) {
    char** current = stringArray;
    while (*current != NULL) {
        free(*current);
        current++;
    }
    free(stringArray);
}
