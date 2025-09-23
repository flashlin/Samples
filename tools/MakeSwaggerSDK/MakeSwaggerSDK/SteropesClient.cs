using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Http;
using Newtonsoft.Json;
using System.ComponentModel.DataAnnotations;

namespace SteropesSDK
{
    /// <summary>
    /// 
    /// </summary>
    public class AutoWithdrawalUser
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("isPassed")]
        public bool? IsPassed { get; set; }

        [JsonProperty("results")]
        public List<BlockAutoWithdrawalResult> Results { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class B2CDuplicateAccountResult
    {
        [JsonProperty("duplicateAccounts")]
        public List<DuplicateAccountForReport> DuplicateAccounts { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class B2CIOMDuplicateAccountRequest
    {
        [JsonProperty("requestId")]
        public string RequestId { get; set; }

        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class B2CIOMDuplicateAccountResponse
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("isPass")]
        public bool? IsPass { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public enum BlockAutoWithdrawalReason
    {
        _0,
        _1,
        _2,
        _4,
        _8,
        _16,
        _32,
        _64,
        _128,
        _256,
        _512,
        _1024,
        _2048
    }

    /// <summary>
    /// 
    /// </summary>
    public class BlockAutoWithdrawalResult
    {
        [JsonProperty("isPassed")]
        public bool? IsPassed { get; set; }

        [JsonProperty("reason")]
        public BlockAutoWithdrawalReason Reason { get; set; }

        [JsonProperty("remark")]
        public string Remark { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public enum BonusHunterBetStatusType
    {
        _7,
        _8,
        _9,
        _10
    }

    /// <summary>
    /// 
    /// </summary>
    public enum BonusHunterCouponStatusType
    {
        _0,
        _1,
        _2,
        _3
    }

    /// <summary>
    /// 
    /// </summary>
    public enum BonusHunterDepositStatusType
    {
        _4,
        _5,
        _6
    }

    /// <summary>
    /// 
    /// </summary>
    public class BonusHunterGroupInfo
    {
        [JsonProperty("totalCount")]
        public int? TotalCount { get; set; }

        [JsonProperty("bonusHunters")]
        public List<BonusHunterGroupV2> BonusHunters { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class BonusHunterGroupV2
    {
        [JsonProperty("accounts")]
        public List<BonusHunterV2> Accounts { get; set; }

        [JsonProperty("groupId")]
        public int? GroupId { get; set; }

        [JsonProperty("isResolved")]
        public bool? IsResolved { get; set; }

        [JsonProperty("canIgnore")]
        public bool? CanIgnore { get; set; }

        [JsonProperty("promotionStatus")]
        public BonusHunterCouponStatusType PromotionStatus { get; set; }

        [JsonProperty("depositStatus")]
        public BonusHunterDepositStatusType DepositStatus { get; set; }

        [JsonProperty("betStatus")]
        public BonusHunterBetStatusType BetStatus { get; set; }

        [JsonProperty("reason")]
        public string Reason { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class BonusHunterResolved
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("remark")]
        public string Remark { get; set; }

        [JsonProperty("modifiedBy")]
        public string ModifiedBy { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public enum BonusHunterType
    {
        _0,
        _1,
        _2,
        _3
    }

    /// <summary>
    /// 
    /// </summary>
    public class BonusHunterV2
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("accountId")]
        public string AccountId { get; set; }

        [JsonProperty("createdOn")]
        public DateTime? CreatedOn { get; set; }

        [JsonProperty("currency")]
        public Currency Currency { get; set; }

        [JsonProperty("currencyStr")]
        public string CurrencyStr { get; set; }

        [JsonProperty("lastLoginIp")]
        public string LastLoginIp { get; set; }

        [JsonProperty("firstName")]
        public string FirstName { get; set; }

        [JsonProperty("surName")]
        public string SurName { get; set; }

        [JsonProperty("dateOfBirth")]
        public DateTime? DateOfBirth { get; set; }

        [JsonProperty("promotionCode")]
        public string PromotionCode { get; set; }

        [JsonProperty("isPromotionDisqualified")]
        public bool? IsPromotionDisqualified { get; set; }

        [JsonProperty("isPromotionCodeEmpty")]
        public bool? IsPromotionCodeEmpty { get; set; }

        [JsonProperty("isPromotionValid")]
        public bool? IsPromotionValid { get; set; }

        [JsonProperty("initialDeposit")]
        public double? InitialDeposit { get; set; }

        [JsonProperty("maxEntitlement")]
        public double? MaxEntitlement { get; set; }

        [JsonProperty("firstTopupDate")]
        public DateTime? FirstTopupDate { get; set; }

        [JsonProperty("firstBetDate")]
        public DateTime? FirstBetDate { get; set; }

        [JsonProperty("actualStake1")]
        public double? ActualStake1 { get; set; }

        [JsonProperty("playerWinlost1")]
        public double? PlayerWinlost1 { get; set; }

        [JsonProperty("actualStake2")]
        public double? ActualStake2 { get; set; }

        [JsonProperty("playerWinlost2")]
        public double? PlayerWinlost2 { get; set; }

        [JsonProperty("isClosed")]
        public bool? IsClosed { get; set; }

        [JsonProperty("remark")]
        public string Remark { get; set; }

        [JsonProperty("modifiedBy")]
        public string ModifiedBy { get; set; }

        [JsonProperty("groupId")]
        public int? GroupId { get; set; }

        [JsonProperty("isResolved")]
        public bool? IsResolved { get; set; }

        [JsonProperty("problem")]
        public string Problem { get; set; }

        [JsonProperty("isUnused")]
        public bool? IsUnused { get; set; }

        [JsonProperty("description")]
        public string Description { get; set; }

        [JsonProperty("isSuspectBH")]
        public bool? IsSuspectBH { get; set; }

        [JsonProperty("isTopupUser")]
        public bool? IsTopupUser { get; set; }

        [JsonProperty("isAbnormalEmail")]
        public bool? IsAbnormalEmail { get; set; }

        [JsonProperty("riskyScore")]
        public double? RiskyScore { get; set; }

        [JsonProperty("whitelistReason")]
        public BonusHunterWhitelistReasonDetail WhitelistReason { get; set; }

        [JsonProperty("isSuspended")]
        public bool? IsSuspended { get; set; }

        [JsonProperty("whitelistReasonStr")]
        public string WhitelistReasonStr { get; set; }

        [JsonProperty("bonusHunterType")]
        public BonusHunterType BonusHunterType { get; set; }

        [JsonProperty("promotionStatus")]
        public BonusHunterCouponStatusType PromotionStatus { get; set; }

        [JsonProperty("depositStatus")]
        public BonusHunterDepositStatusType DepositStatus { get; set; }

        [JsonProperty("betStatus")]
        public BonusHunterBetStatusType BetStatus { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public enum BonusHunterWhitelistReasonDetail
    {
        _0,
        _1,
        _2,
        _3,
        _4
    }

    /// <summary>
    /// 
    /// </summary>
    public class CasinoPromotionHunter
    {
        [JsonProperty("requestId")]
        public string RequestId { get; set; }

        [JsonProperty("interval")]
        public ClosedOpenInterval Interval { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class CasinoPromotionHunterResponse
    {
        [JsonProperty("accountId")]
        public List<string> AccountId { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class CasinoPromotionHunterResponseV2
    {
        [JsonProperty("pairedBettingGroup")]
        public List<PairedBettingGroup> PairedBettingGroup { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class ChargeBackRequest
    {
        [JsonProperty("requestId")]
        public string RequestId { get; set; }

        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class ChargeBackResponse
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("isProblematic")]
        public bool? IsProblematic { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class CheckDuplicateAccountRequest
    {
        [JsonProperty("usernames")]
        public List<string> Usernames { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class CheckDuplicateRequest
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("uplineCustomerIds")]
        public List<int> UplineCustomerIds { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class CheckDuplicateResponse
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("isDuplicate")]
        public bool? IsDuplicate { get; set; }

        [JsonProperty("duplicateAccountIds")]
        public List<int> DuplicateAccountIds { get; set; }

        [JsonProperty("isSuccess")]
        public bool? IsSuccess { get; set; }

        [JsonProperty("errorMessage")]
        public string ErrorMessage { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class ClosedOpenInterval
    {
        [JsonProperty("from")]
        public DateTime? From { get; set; }

        [JsonProperty("to")]
        public DateTime? To { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public enum Currency
    {
        _0,
        _1,
        _2,
        _3,
        _4,
        _5,
        _6,
        _7,
        _9,
        _12,
        _13,
        _25,
        _26,
        _29,
        _30,
        _32,
        _33,
        _34,
        _35,
        _36,
        _37,
        _38,
        _39,
        _40,
        _41,
        _42,
        _43,
        _44,
        _45,
        _46,
        _47,
        _48,
        _49,
        _50,
        _51,
        _52
    }

    /// <summary>
    /// 
    /// </summary>
    public class CustomerProductPreference
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("product")]
        public string Product { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class DenyRebateComplianceRequest
    {
        [JsonProperty("requestId")]
        public string RequestId { get; set; }

        [JsonProperty("customerIds")]
        public List<int> CustomerIds { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class DuplicateAccountForReport
    {
        [JsonProperty("accountId")]
        public string AccountId { get; set; }

        [JsonProperty("currency")]
        public string Currency { get; set; }

        [JsonProperty("isClosed")]
        public bool? IsClosed { get; set; }

        [JsonProperty("duplicateAccounts")]
        public List<DuplicateAccountInfoForReport> DuplicateAccounts { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class DuplicateAccountInfoForReport
    {
        [JsonProperty("accountId")]
        public string AccountId { get; set; }

        [JsonProperty("linkedReason")]
        public string LinkedReason { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class DuplicateAccountInfoResult
    {
        [JsonProperty("sourceUsername")]
        public string SourceUsername { get; set; }

        [JsonProperty("duplicateUsername")]
        public List<string> DuplicateUsername { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class FraudDetectionResponse
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("isValid")]
        public bool? IsValid { get; set; }

        [JsonProperty("reasonCategory")]
        public string ReasonCategory { get; set; }

        [JsonProperty("reasonDetail")]
        public string ReasonDetail { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class GamesRecommendationResult
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("recommendedGames")]
        public List<int> RecommendedGames { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class NotifyCustomerRewardedRequest
    {
        [JsonProperty("custId")]
        public int? CustId { get; set; }

        [JsonProperty("promotionCode")]
        public string PromotionCode { get; set; }

        [JsonProperty("rewardTypes")]
        public List<PromotionRewardType> RewardTypes { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class PairedBettingGroup
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("pairedCustomerId")]
        public int? PairedCustomerId { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class PlayerAbnormalGame
    {
        [JsonProperty("accountId")]
        public string AccountId { get; set; }

        [JsonProperty("gameCode")]
        public int? GameCode { get; set; }

        [JsonProperty("reason")]
        public string Reason { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class PromotionCheckResponse
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("isSuggestCancel")]
        public bool? IsSuggestCancel { get; set; }

        [JsonProperty("remark")]
        public string Remark { get; set; }

        [JsonProperty("actionTaker")]
        public string ActionTaker { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public enum PromotionRewardType
    {
        _0,
        _1,
        _2,
        _3,
        _4,
        _7
    }

    /// <summary>
    /// 
    /// </summary>
    public class RebateComplianceRequest
    {
        [JsonProperty("requestId")]
        public string RequestId { get; set; }

        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class RemoveBonusHunterWhitelistRequest
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class RiskyScoreReason
    {
        [JsonProperty("reason")]
        public string Reason { get; set; }

        [JsonProperty("remark")]
        public string Remark { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class RiskyScoreRequest
    {
        [JsonProperty("customerIds")]
        public List<int> CustomerIds { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class RiskyScoreResult
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("isPass")]
        public bool? IsPass { get; set; }

        [JsonProperty("reasons")]
        public List<RiskyScoreReason> Reasons { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class VoucherComplianceCheckResponse
    {
        [JsonProperty("customerId")]
        public int? CustomerId { get; set; }

        [JsonProperty("isPassed")]
        public bool? IsPassed { get; set; }

        [JsonProperty("reasonFlag")]
        public int? ReasonFlag { get; set; }

        [JsonProperty("remark")]
        public string Remark { get; set; }

        [JsonProperty("product")]
        public string Product { get; set; }

    }

    /// <summary>
    /// HTTP client for Steropes API
    /// </summary>
    public class SteropesClient
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        /// <summary>
        /// Initializes a new instance of SteropesClient
        /// </summary>
        /// <param name="httpClientFactory">HTTP client factory</param>
        /// <param name="baseUrl">Base URL for the API</param>
        public SteropesClient(IHttpClientFactory httpClientFactory, string baseUrl)
        {
            _httpClient = httpClientFactory.CreateClient();
            _baseUrl = baseUrl.TrimEnd('/');
        }

        /// <summary>
        /// Initializes a new instance of SteropesClient
        /// </summary>
        /// <param name="httpClient">HTTP client instance</param>
        /// <param name="baseUrl">Base URL for the API</param>
        public SteropesClient(HttpClient httpClient, string baseUrl)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _baseUrl = baseUrl.TrimEnd('/');
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="accountId"></param>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns>Task</returns>
        public async Task CheckByAccountId(string? accountId = null, DateTime? start = null, DateTime? end = null)
        {
            var url = $"/v1/AbnormalGamePlayer/CheckByAccountId";
            var queryParams = new List<string>();
            if (accountId != null)
                queryParams.Add($"accountId={Uri.EscapeDataString(accountId.ToString())}");
            if (start != null)
                queryParams.Add($"start={Uri.EscapeDataString(start.ToString())}");
            if (end != null)
                queryParams.Add($"end={Uri.EscapeDataString(end.ToString())}");
            if (queryParams.Any())
                url += "?" + string.Join("&", queryParams);

            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="gamecode"></param>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns>Task</returns>
        public async Task CheckByGame(int? gamecode = null, DateTime? start = null, DateTime? end = null)
        {
            var url = $"/v1/AbnormalGamePlayer/CheckByGame";
            var queryParams = new List<string>();
            if (gamecode != null)
                queryParams.Add($"gamecode={Uri.EscapeDataString(gamecode.ToString())}");
            if (start != null)
                queryParams.Add($"start={Uri.EscapeDataString(start.ToString())}");
            if (end != null)
                queryParams.Add($"end={Uri.EscapeDataString(end.ToString())}");
            if (queryParams.Any())
                url += "?" + string.Join("&", queryParams);

            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="prefix"></param>
        /// <returns>Task</returns>
        public async Task ReloadCache(string? prefix = null)
        {
            var url = $"/v1/AdHoc/ReloadCache";
            var queryParams = new List<string>();
            if (prefix != null)
                queryParams.Add($"prefix={Uri.EscapeDataString(prefix.ToString())}");
            if (queryParams.Any())
                url += "?" + string.Join("&", queryParams);

            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task ReloadVoucherComplianceCache(int body)
        {
            var url = $"/v1/AdHoc/ReloadVoucherComplianceCache";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="customerId"></param>
        /// <returns>Task</returns>
        public async Task GetProductPreference(int customerId)
        {
            var url = $"/v1/AutoWithdrawal/getProductPreference/{customerId}";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="customerId"></param>
        /// <returns>Task</returns>
        public async Task Check(int customerId)
        {
            var url = $"/v1/AutoWithdrawal/check/{customerId}";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task Casinopromotionhunter(ClosedOpenInterval body)
        {
            var url = $"/v1/BetScan/casinopromotionhunter";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task Casinopromotionhunter(CasinoPromotionHunter body)
        {
            var url = $"/v2/BetScanControllerV2/casinopromotionhunter";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="accountId"></param>
        /// <returns>Task</returns>
        public async Task BonusHunterForBDT(string accountId)
        {
            var url = $"/v1/BonusHunter/BonusHunterForBDT/{accountId}";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task UpdateBonusHunterRemark(BonusHunterResolved body)
        {
            var url = $"/v1/BonusHunter/UpdateBonusHunterRemark";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task CheckRiskyScore(RiskyScoreRequest body)
        {
            var url = $"/v1/BonusHunter/CheckRiskyScore";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task CheckC2CDuplicateAccount(List<CheckDuplicateRequest> body)
        {
            var url = $"/v1/C2C/checkC2CDuplicateAccount";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task DepositPredict(ChargeBackRequest body)
        {
            var url = $"/v1/ChargeBack/depositPredict";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task WithdrawPredict(ChargeBackRequest body)
        {
            var url = $"/v1/ChargeBack/withdrawPredict";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task Removebonushunterwhitelist(RemoveBonusHunterWhitelistRequest body)
        {
            var url = $"/v1/Compliance/removebonushunterwhitelist";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="customerId"></param>
        /// <returns>Task</returns>
        public async Task CheckBonusHunterForBDT(int customerId)
        {
            var url = $"/v1/Compliance/CheckBonusHunterForBDT/{customerId}";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="customerId"></param>
        /// <returns>Task</returns>
        public async Task CheckBDT(int customerId)
        {
            var url = $"/v2/DailyMissionCompliance/CheckBDT/{customerId}";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task Check(int body)
        {
            var url = $"/v2/DailyMissionCompliance/Check";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task Check(DenyRebateComplianceRequest body)
        {
            var url = $"/v1/DenyRebate/Check";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task CheckB2CAccount(CheckDuplicateAccountRequest body)
        {
            var url = $"/v1/DuplicateAccount/CheckB2CAccount";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task CheckB2BAccount(CheckDuplicateAccountRequest body)
        {
            var url = $"/v1/DuplicateAccount/CheckB2BAccount";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task CheckB2CIOM(B2CIOMDuplicateAccountRequest body)
        {
            var url = $"/v1/DuplicateAccount/CheckB2CIOM";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="custId"></param>
        /// <param name="requestId"></param>
        /// <returns>Task</returns>
        public async Task Check(int? custId = null, string? requestId = null)
        {
            var url = $"/v1/ExtraVoucherCompliance/Check";
            var queryParams = new List<string>();
            if (custId != null)
                queryParams.Add($"custId={Uri.EscapeDataString(custId.ToString())}");
            if (requestId != null)
                queryParams.Add($"requestId={Uri.EscapeDataString(requestId.ToString())}");
            if (queryParams.Any())
                url += "?" + string.Join("&", queryParams);

            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>Task</returns>
        public async Task Check()
        {
            var url = $"/v1/FraudDetection/Check";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>Task</returns>
        public async Task BankSlip()
        {
            var url = $"/v1/OCR/BankSlip";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>Task</returns>
        public async Task TestGemini()
        {
            var url = $"/v1/OCR/TestGemini";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task Check(RebateComplianceRequest body)
        {
            var url = $"/v1/RebateCompliance/Check";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>Task</returns>
        public async Task Games()
        {
            var url = $"/v1/Recommendation/Games";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>Task</returns>
        public async Task Env()
        {
            var url = $"/v1/console/env";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task CheckCustomerPromotions(int body)
        {
            var url = $"/v1/Test/CheckCustomerPromotions";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task CheckCustomerPromotionStatus(int body)
        {
            var url = $"/v1/Test/CheckCustomerPromotionStatus";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task Notifycustomerrewarded(NotifyCustomerRewardedRequest body)
        {
            var url = $"/v1/UserScan/notifycustomerrewarded";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="body">Request body</param>
        /// <returns>Task</returns>
        public async Task NewNotifycustomerrewarded(NotifyCustomerRewardedRequest body)
        {
            var url = $"/v1/UserScan/NewNotifycustomerrewarded";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            if (body != null)
            {
                var jsonContent = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.StatusCode}: {errorContent}");
            }

            // No return value expected
        }

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            _httpClient?.Dispose();
        }

    }
}
