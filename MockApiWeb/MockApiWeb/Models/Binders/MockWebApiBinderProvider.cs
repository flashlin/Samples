using Microsoft.AspNetCore.Mvc.ModelBinding;
using MockApiWeb.Models.Binders;
using MockApiWeb.Models.Requests;

namespace MockApiWeb.Models.Middlewares;

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