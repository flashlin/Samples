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
		private static readonly ConcurrentDictionary<string, string[]> _varyByParamsSplitCache = new ConcurrentDictionary<string, string[]>();
		private static readonly MemoryCache _cache = new MemoryCache("ActionResultCacheAttribute");

		public string VaryByParam { get; set; }
		public int Duration { get; set; }

		public override void OnActionExecuting(ActionExecutingContext filterContext)
		{
			var cacheKey = CreateCacheKey(filterContext.RouteData.Values, filterContext.ActionParameters);

			var result = _cache.Get(cacheKey) as ActionResult;
			if (result != null)
			{
				filterContext.Result = result;
				return;
			}

			// Store to HttpContext Items
			filterContext.HttpContext.Items["__actionresultcacheattribute_cachekey"] = cacheKey;
		}

		public override void OnActionExecuted(ActionExecutedContext filterContext)
		{
			if (filterContext.Exception != null)
			{
				return;
			}

			// Get the cache key from HttpContext Items
			var cacheKey = filterContext.HttpContext.Items["__actionresultcacheattribute_cachekey"] as string;
			if (string.IsNullOrWhiteSpace(cacheKey))
			{
				return;
			}

			if (Duration != 0)
			{
				_cache.Add(cacheKey, filterContext.Result, DateTime.UtcNow.AddSeconds(Duration));
				return;
			}

			_cache.Add(cacheKey, filterContext.Result, DateTime.UtcNow.AddSeconds(60 * 60));
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

			if (!_varyByParamsSplitCache.TryGetValue(VaryByParam, out var varyByParamsSplit))
			{
				varyByParamsSplit = VaryByParam.Split(new[] { ',', ' ' },
					StringSplitOptions.RemoveEmptyEntries);
				_varyByParamsSplitCache[VaryByParam] = varyByParamsSplit;
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