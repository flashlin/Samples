using Reqnroll;

namespace TestProject
{
    [Binding]
    public class ReqnrollHooks
    {
        [BeforeTestRun]
        public static void BeforeTestRun()
        {
            // Initialize any test run setup here
        }

        [AfterTestRun]
        public static void AfterTestRun()
        {
            // Clean up any test run resources here
        }
    }
} 