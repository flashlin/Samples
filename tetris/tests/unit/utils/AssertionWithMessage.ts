class JestAssertionError extends Error {
   matcherResult: any;

   constructor(result: any, callsite: any) {
      super(result.message());
      this.matcherResult = result;

      if (Error.captureStackTrace) {
         Error.captureStackTrace(this, callsite);
      }
   }
}

const wrapMatcher = (matcher: any, customMessage: any) => {
   const newMatcher = (...args: any) => {
      try {
         return matcher(...args);
      } catch (error) {
         if (!error.matcherResult) {
            throw error;
         }
         const { matcherResult } = error;

         if (typeof customMessage !== 'string' || customMessage.length < 1) {
            throw new JestAssertionError(matcherResult, newMatcher);
         }

         const message = () =>
            'Custom message:\n  ' +
            customMessage +
            '\n\n' +
            matcherResult.message();

         throw new JestAssertionError(
            { ...matcherResult, message },
            newMatcher
         );
      }
   };
   return newMatcher;
};

const wrapMatchers: any = (matchers: any, customMessage: any) => {
   return Object.keys(matchers).reduce((acc, name) => {
      const matcher = matchers[name];

      if (typeof matcher === 'function') {
         return {
            ...acc,
            [name]: wrapMatcher(matcher, customMessage)
         };
      }

      return {
         ...acc,
         [name]: wrapMatchers(matcher, customMessage) // recurse on .not/.resolves/.rejects
      };
   }, {});
};

export default (expect: any) => {
   // proxy the expect function
   let expectProxy = Object.assign(
      // partially apply expect to get all matchers and chain them
      (actual: any, customMessage: any) =>
         wrapMatchers(expect(actual), customMessage),
      expect // clone additional properties on expect
   );

   expectProxy.extend = (o: any) => {
      expect.extend(o); // add new matchers to expect
      expectProxy = Object.assign(expectProxy, expect); // clone new asymmetric matchers
   };

   return expectProxy;
};

