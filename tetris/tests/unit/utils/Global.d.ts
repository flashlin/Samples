declare module NodeJS  {
   interface Global {
      expect: any;
   }
}

declare module jest  {
   interface Expect {
      <T=any>(actual: T, desc?: string) : jest.Matchers<T>;
   }
}