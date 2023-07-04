export function MockMethod(mockRun: boolean, returnValue?: any) {
    return function (
        target: any,
        propertyKey: string,
        descriptor: PropertyDescriptor
    ) {
        const originalMethod = descriptor.value;
        descriptor.value = function (...args: any[]) {
            //console.log(`Calling ${propertyKey} with arguments: ${JSON.stringify(args)}`);
            if (mockRun == true) {
                return returnValue;
            }
            const result = originalMethod.apply(this, args);
            //console.log(`Return value of ${propertyKey}: ${JSON.stringify(result)}`);
            return result;
        };
        return descriptor;
    };
}


export function MockAsyncMethod(mockRun: boolean, returnValue?: any) {
    return MockMethod(mockRun, Promise.resolve(returnValue));
}


export function MockFuncCall(mockRun: boolean, mockFn: (...args: any[]) => any, fn: (...args: any[]) => any) {
    return function (...args: any[]) {
        if( mockRun ) {
            return mockFn(...args);
        }
        return fn(...args);
    };
}