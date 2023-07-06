export type AnyFunc = (...args: any[]) => any;

type MethodDecoratorWithArgs<T extends AnyFunc> = (target: any,
    propertyKey: string | symbol, descriptor: TypedPropertyDescriptor<T>) =>
    TypedPropertyDescriptor<T> | void;


// export function MockMethod<T, U extends AnyFunc>(mockRun: boolean, returnValue: Function|object|null): MethodDecoratorWithArgs<U> {
//     return function (
//         target: any,
//         propertyKey: string | symbol,
//         descriptor: TypedPropertyDescriptor<U>
//     ) {
//         const originalMethod = descriptor.value;
//         descriptor.value = function (this: T, ...args: any[]) {
//             //console.log(`Calling ${propertyKey} with arguments: ${JSON.stringify(args)}`);
//             if (mockRun == true) {
//                 if (typeof returnValue === 'function') {
//                     return (returnValue as Function).apply(this, args);
//                 }
//                 return returnValue;
//             }
//             if (originalMethod && typeof originalMethod === 'function') {
//                 const result = originalMethod.apply(this, args);
//                 //console.log(`Return value of ${propertyKey}: ${JSON.stringify(result)}`);
//                 return result;
//             }
//         } as unknown as U;
//         return descriptor;
//     };
// }

export function MockMethod(mockRun: boolean, returnValue: AnyFunc | Object | null) {
    return function (
        target: any,
        propertyKey: string,
        descriptor: PropertyDescriptor
    ) {
        const originalMethod = descriptor.value;
        descriptor.value = function (...args: any[]) {
            //console.log(`Calling ${propertyKey} with arguments: ${JSON.stringify(args)}`);
            if (mockRun == true) {
                if( typeof returnValue == "function") {
                    return (returnValue as Function).apply(this, args);
                }
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


export function MockFuncCall<TReturn>(mockRun:
    boolean, mockFn: () => TReturn,
    fn: () => TReturn):
    () => TReturn;

export function MockFuncCall<T0, TReturn>(mockRun:
    boolean, mockFn: (args: T0) => TReturn,
    fn: (args: T0) => TReturn):
    (args: T0) => TReturn;

export function MockFuncCall<T0, T1, TReturn>(
    mockRun: boolean,
    mockFn: (args0: T0, args1: T1) => TReturn,
    fn: (args0: T0, args1: T1) => TReturn):
    (args0: T0, args1: T1) => TReturn;

export function MockFuncCall<T0, T1, T2, TReturn>(
    mockRun: boolean,
    mockFn: (args0: T0, args1: T1, args2: T2) => TReturn,
    fn: (args0: T0, args1: T1, args2: T2) => TReturn):
    (args0: T0, args1: T1, args2: T2) => TReturn;

export function MockFuncCall<T0, T1, T2, T3, TReturn>(
    mockRun: boolean,
    mockFn: (args0: T0, args1: T1, args2: T2, args3: T3) => TReturn,
    fn: (args0: T0, args1: T1, args2: T2, args3: T3) => TReturn):
    (args0: T0, args1: T1, args2: T2, args3: T3) => TReturn;


export function MockFuncCall<T0, T1, T2, T3, T4, TReturn>(
    mockRun: boolean,
    mockFn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4) => TReturn,
    fn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4) => TReturn):
    (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4) => TReturn;


export function MockFuncCall<T0, T1, T2, T3, T4, T5, TReturn>(
    mockRun: boolean,
    mockFn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5) => TReturn,
    fn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5) => TReturn):
    (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5) => TReturn;


export function MockFuncCall<T0, T1, T2, T3, T4, T5, T6, TReturn>(
    mockRun: boolean,
    mockFn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5, args6: T6) => TReturn,
    fn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5, args6: T6) => TReturn):
    (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5, args6: T6) => TReturn;


export function MockFuncCall<T0, T1, T2, T3, T4, T5, T6, T7, TReturn>(
    mockRun: boolean,
    mockFn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5, args6: T6, args7: T7) => TReturn,
    fn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5, args6: T6, args7: T7) => TReturn):
    (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5, args6: T6, args7: T7) => TReturn;


export function MockFuncCall<T0, T1, T2, T3, T4, T5, T6, T7, T8, TReturn>(
    mockRun: boolean,
    mockFn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5, args6: T6, args7: T7, args8: T8) => TReturn,
    fn: (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5, args6: T6, args7: T7, args8: T8) => TReturn):
    (args0: T0, args1: T1, args2: T2, args3: T3, args4: T4, args5: T5, args6: T6, args7: T7, args8: T8) => TReturn;


export function MockFuncCall(mockRun: boolean, mockFn: (...args: any[]) => any, fn: (...args: any[]) => any) {
    return function (...args: any[]) {
        if (mockRun) {
            return mockFn(...args);
        }
        return fn(...args);
    };
}

