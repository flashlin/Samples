export function MockAsyncMethod(returnValue?: string) {
    return function(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        const originalMethod = descriptor.value;
        descriptor.value = function (...args: any[]) {
            console.log(`Calling ${propertyKey} with arguments: ${JSON.stringify(args)}`);
            const mode = import.meta.env.MODE;
            if( mode == 'development' ) {
                if( returnValue == null ) {
                    returnValue = "{}";
                }
                return Promise.resolve(JSON.parse(returnValue));
            }
            const result = originalMethod.apply(this, args);
            console.log(`Return value of ${propertyKey}: ${JSON.stringify(result)}`);
            return result;
        };
        return descriptor;
    };
}
  