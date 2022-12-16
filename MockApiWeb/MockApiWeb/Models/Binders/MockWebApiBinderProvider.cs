using Microsoft.AspNetCore.Mvc.ModelBinding;
using MockApiWeb.Models.DataConstraints;

namespace MockApiWeb.Models.Binders;

public class MockWebApiBinderProvider : IModelBinderProvider
{
    public IModelBinder? GetBinder(ModelBinderProviderContext context)
    {
        if (context.Metadata.ModelType == typeof(MockWebApiRequest))
        {
            return context.Services.GetRequiredService<MockWebApiRequestBinder>();
        }
        return null;
    }
}