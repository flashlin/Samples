using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Caching;
using System.Text;
using System.Web;
using System.Web.Mvc;
using System.Web.Routing;

namespace Custom_AspNet_Mvc_ActionResult_CacheAttribute.Models
{
	public class ActionResultCacheAttribute : ActionFilterAttribute
	{
		private const string ActionresultcacheattributeCachekey = "__actionresultcacheattribute_cachekey";
		private static readonly ConcurrentDictionary<string, string[]> VaryByParamsSplitCache = new ConcurrentDictionary<string, string[]>();
		private static readonly MemoryCache Cache = new MemoryCache("ActionResultCacheAttribute");

		public string VaryByParam { get; set; }
		public int Duration { get; set; }

		public override void OnActionExecuting(ActionExecutingContext filterContext)
		{
			var cacheKey = CreateCacheKey(filterContext.RouteData.Values, filterContext.ActionParameters);

			if (Cache.Get(cacheKey) is ActionResult result)
			{
				filterContext.Result = result;
				return;
			}

			filterContext.HttpContext.Items[ActionresultcacheattributeCachekey] = cacheKey;
		}

		public override void OnActionExecuted(ActionExecutedContext filterContext)
		{
			if (filterContext.Exception != null)
			{
				return;
			}

			// Get the cache key from HttpContext Items
			var cacheKey = filterContext.HttpContext.Items[ActionresultcacheattributeCachekey] as string;
			if (string.IsNullOrWhiteSpace(cacheKey))
			{
				return;
			}

			if (Duration != 0)
			{
				Cache.Add(cacheKey, filterContext.Result, DateTime.UtcNow.AddSeconds(Duration));
				return;
			}

			Cache.Add(cacheKey, filterContext.Result, DateTime.UtcNow.AddSeconds(60 * 60));
		}

		private string CreateCacheKey(RouteValueDictionary routeValues,
			IDictionary<string, object> actionParameters)
		{
			var sb = new StringBuilder(routeValues["controller"].ToString());
			sb.Append("_").Append(routeValues["action"].ToString());

			if (string.IsNullOrWhiteSpace(VaryByParam))
			{
				return sb.ToString();
			}

			if (VaryByParam == "*")
			{
				foreach (var p in actionParameters)
				{
					sb.Append("_");
					sb.Append(GetObjectValue(p.Value));
				}
				return sb.ToString();
			}

			if (!VaryByParamsSplitCache.TryGetValue(VaryByParam, out var varyByParamsSplit))
			{
				varyByParamsSplit = VaryByParam.Split(new[] { ',', ' ' },
					StringSplitOptions.RemoveEmptyEntries);
				VaryByParamsSplitCache[VaryByParam] = varyByParamsSplit;
			}

			foreach (var varyByParam in varyByParamsSplit)
			{
				if (!actionParameters.TryGetValue(varyByParam, out object varyByParamObject))
				{
					continue;
				}
				sb.Append("_");
				sb.Append(GetObjectValue(varyByParamObject));
			}

			return sb.ToString();
		}

		protected virtual string GetObjectValue(object obj)
		{
			if (obj == null)
			{
				return "{null}";
			}
			return obj.ToString();
		}
	}
}