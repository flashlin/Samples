using Microsoft.EntityFrameworkCore;

namespace Generated
{
    public partial class AccountDBDbContext : DbContext
    {
        public DbSet<__RefactorLogEntity> __RefactorLog { get; set; }
        public DbSet<AccountActionLogEntity> AccountActionLog { get; set; }
        public DbSet<AccountingCheckSumEntity> AccountingCheckSum { get; set; }
        public DbSet<AccountingDateEntity> AccountingDate { get; set; }
        public DbSet<AccountSettleBetTempEntity> AccountSettleBetTemp { get; set; }
        public DbSet<AccountSuspendLimitEntity> AccountSuspendLimit { get; set; }
        public DbSet<AgentFeatureEntity> AgentFeature { get; set; }
        public DbSet<AgentMappingEntity> AgentMapping { get; set; }
        public DbSet<AgentPatternLoginEntity> AgentPatternLogin { get; set; }
        public DbSet<AgentsBetSettingEntity> AgentsBetSetting { get; set; }
        public DbSet<AgentsDiscountEntity> AgentsDiscount { get; set; }
        public DbSet<AgentsMaxPTEntity> AgentsMaxPT { get; set; }
        public DbSet<AgentsMonitorListEntity> AgentsMonitorList { get; set; }
        public DbSet<AgentsTransferSettingEntity> AgentsTransferSetting { get; set; }
        public DbSet<APIAccountEntity> APIAccount { get; set; }
        public DbSet<AuditCheckedListEntity> AuditCheckedList { get; set; }
        public DbSet<AuditGroupMoveLogEntity> AuditGroupMoveLog { get; set; }
        public DbSet<AutoCBSettingEntity> AutoCBSetting { get; set; }
        public DbSet<AutoSettleLeagueListEntity> AutoSettleLeagueList { get; set; }
        public DbSet<AutoSettlementEntity> AutoSettlement { get; set; }
        public DbSet<AutoSettlementRecordEntity> AutoSettlementRecord { get; set; }
        public DbSet<AutoSuspendLogEntity> AutoSuspendLog { get; set; }
        public DbSet<B2B2CRebateReportEntity> B2B2CRebateReport { get; set; }
        public DbSet<BankAccountEntity> BankAccount { get; set; }
        public DbSet<BankGroupBankInfoEntity> BankGroupBankInfo { get; set; }
        public DbSet<BankGroupMoneyChangerSenderInfoEntity> BankGroupMoneyChangerSenderInfo { get; set; }
        public DbSet<BankGroupPartnershipSenderInfoEntity> BankGroupPartnershipSenderInfo { get; set; }
        public DbSet<BatchLockEntity> BatchLock { get; set; }
        public DbSet<BatchResultHistoryEntity> BatchResultHistory { get; set; }
        public DbSet<BetBuilderLegEntity> BetBuilderLeg { get; set; }
        public DbSet<BetBuilderLeg14Entity> BetBuilderLeg14 { get; set; }
        public DbSet<bettransEntity> bettrans { get; set; }
        public DbSet<bettrans14Entity> bettrans14 { get; set; }
        public DbSet<bettransmEntity> bettransm { get; set; }
        public DbSet<bettransm14Entity> bettransm14 { get; set; }
        public DbSet<BlackListWordEntity> BlackListWord { get; set; }
        public DbSet<BlindRiskCustomerEntity> BlindRiskCustomer { get; set; }
        public DbSet<BlindRiskCustomerClusterEntity> BlindRiskCustomerCluster { get; set; }
        public DbSet<BlindRiskCustomerDailyBetSummaryEntity> BlindRiskCustomerDailyBetSummary { get; set; }
        public DbSet<BonusTransactionsEntity> BonusTransactions { get; set; }
        public DbSet<BonusWalletEntity> BonusWallet { get; set; }
        public DbSet<BonusWalletCashSettledEntity> BonusWalletCashSettled { get; set; }
        public DbSet<BonusWalletCashUsedEntity> BonusWalletCashUsed { get; set; }
        public DbSet<BonusWalletDailyStatementEntity> BonusWalletDailyStatement { get; set; }
        public DbSet<BonusWalletStatementEntity> BonusWalletStatement { get; set; }
        public DbSet<CancelledMatchEntity> CancelledMatch { get; set; }
        public DbSet<CashOutActionLogEntity> CashOutActionLog { get; set; }
        public DbSet<CashOutBettransEntity> CashOutBettrans { get; set; }
        public DbSet<CashOutBettransmEntity> CashOutBettransm { get; set; }
        public DbSet<CashSettledEntity> CashSettled { get; set; }
        public DbSet<CashSettledLogEntity> CashSettledLog { get; set; }
        public DbSet<CashSettledLogSumEntity> CashSettledLogSum { get; set; }
        public DbSet<CashSettledTransactionEntity> CashSettledTransaction { get; set; }
        public DbSet<CashSettledUpsertEntity> CashSettledUpsert { get; set; }
        public DbSet<CashUsedEntity> CashUsed { get; set; }
        public DbSet<CashUsedLogEntity> CashUsedLog { get; set; }
        public DbSet<CashUsedLogSumEntity> CashUsedLogSum { get; set; }
        public DbSet<ChangeParentLogEntity> ChangeParentLog { get; set; }
        public DbSet<CitiesEntity> Cities { get; set; }
        public DbSet<CompanyRiskEntity> CompanyRisk { get; set; }
        public DbSet<CorporateCustomerLogEntity> CorporateCustomerLog { get; set; }
        public DbSet<CountriesEntity> Countries { get; set; }
        public DbSet<CountryPhoneCodeEntity> CountryPhoneCode { get; set; }
        public DbSet<CoverBetAccountEntity> CoverBetAccount { get; set; }
        public DbSet<CurrencyTransactionsEntity> CurrencyTransactions { get; set; }
        public DbSet<custcontrolEntity> custcontrol { get; set; }
        public DbSet<customerEntity> customer { get; set; }
        public DbSet<CustomerCreditEntity> CustomerCredit { get; set; }
        public DbSet<CustomerGroupEntity> CustomerGroup { get; set; }
        public DbSet<CustomerGroupCreditSettingEntity> CustomerGroupCreditSetting { get; set; }
        public DbSet<CustomerMappingEntity> CustomerMapping { get; set; }
        public DbSet<CustomerPromotionEntity> CustomerPromotion { get; set; }
        public DbSet<CustomerPromotionConfigEntity> CustomerPromotionConfig { get; set; }
        public DbSet<CustomerPromotionSignUpEntity> CustomerPromotionSignUp { get; set; }
        public DbSet<CustomerRiskLimitEntity> CustomerRiskLimit { get; set; }
        public DbSet<CustomerSettingVerificationLogEntity> CustomerSettingVerificationLog { get; set; }
        public DbSet<CustomerTournamentEntity> CustomerTournament { get; set; }
        public DbSet<CustomerTrackingEntity> CustomerTracking { get; set; }
        public DbSet<CustomerVvipGroupEntity> CustomerVvipGroup { get; set; }
        public DbSet<DailyCBEntity> DailyCB { get; set; }
        public DbSet<DailyStatementEntity> DailyStatement { get; set; }
        public DbSet<DailyStatementSumEntity> DailyStatementSum { get; set; }
        public DbSet<DailyStatementTransactionEntity> DailyStatementTransaction { get; set; }
        public DbSet<DailyStatementUpsertsEntity> DailyStatementUpserts { get; set; }
        public DbSet<DBActionLogEntity> DBActionLog { get; set; }
        public DbSet<DBErrorLogEntity> DBErrorLog { get; set; }
        public DbSet<DeletedCustomerEntity> DeletedCustomer { get; set; }
        public DbSet<DeletedStmtLogEntity> DeletedStmtLog { get; set; }
        public DbSet<DirectCustomerAuditEntity> DirectCustomerAudit { get; set; }
        public DbSet<DisplayNamePrefixEntity> DisplayNamePrefix { get; set; }
        public DbSet<EarlySettlementEntity> EarlySettlement { get; set; }
        public DbSet<EffectivePostionTakingEntity> EffectivePostionTaking { get; set; }
        public DbSet<EnumEntity> _Enum { get; set; }
        public DbSet<exchangeEntity> exchange { get; set; }
        public DbSet<ExchangeCodeLookUpEntity> ExchangeCodeLookUp { get; set; }
        public DbSet<ExternalExchangeEntity> ExternalExchange { get; set; }
        public DbSet<FailedSettlementBetsEntity> FailedSettlementBets { get; set; }
        public DbSet<FantasyDailyWinLostEntity> FantasyDailyWinLost { get; set; }
        public DbSet<FantasyRankingEntity> FantasyRanking { get; set; }
        public DbSet<FantasySportsBetEntity> FantasySportsBet { get; set; }
        public DbSet<FantasySportsOrderEntity> FantasySportsOrder { get; set; }
        public DbSet<FollowBetAccountEntity> FollowBetAccount { get; set; }
        public DbSet<GamePreOddsEntity> GamePreOdds { get; set; }
        public DbSet<GamesCashUsedTransactionEntity> GamesCashUsedTransaction { get; set; }
        public DbSet<IomCustomersAllowedForSBOProductsEntity> IomCustomersAllowedForSBOProducts { get; set; }
        public DbSet<JoinNowPromotionEntity> JoinNowPromotion { get; set; }
        public DbSet<JoinNowPromotionLogEntity> JoinNowPromotionLog { get; set; }
        public DbSet<KYCEmailNotifyEntity> KYCEmailNotify { get; set; }
        public DbSet<LockAccountEntity> LockAccount { get; set; }
        public DbSet<LoginBlackListEntity> LoginBlackList { get; set; }
        public DbSet<LoginWhiteListEntity> LoginWhiteList { get; set; }
        public DbSet<LuckyBetEntity> LuckyBet { get; set; }
        public DbSet<matchEntity> match { get; set; }
        public DbSet<match_buffertableEntity> match_buffertable { get; set; }
        public DbSet<match14Entity> match14 { get; set; }
        public DbSet<MatchChangeLogEntity> MatchChangeLog { get; set; }
        public DbSet<MatchResultEntity> MatchResult { get; set; }
        public DbSet<MatchResult_14_BufferTableEntity> MatchResult_14_BufferTable { get; set; }
        public DbSet<MatchResult_BufferTableEntity> MatchResult_BufferTable { get; set; }
        public DbSet<MatchResult14Entity> MatchResult14 { get; set; }
        public DbSet<MatchResult14_BufferTableEntity> MatchResult14_BufferTable { get; set; }
        public DbSet<MaxBetReducedAgentsPlayersIdEntity> MaxBetReducedAgentsPlayersId { get; set; }
        public DbSet<MaxPTConfigEntity> MaxPTConfig { get; set; }
        public DbSet<MemberBetSettingEntity> MemberBetSetting { get; set; }
        public DbSet<MemberProdSettingEntity> MemberProdSetting { get; set; }
        public DbSet<MnlStatementForIomCustomerEntity> MnlStatementForIomCustomer { get; set; }
        public DbSet<MnlStatementForIomCustomerRefEntity> MnlStatementForIomCustomerRef { get; set; }
        public DbSet<MoneyChangerTracInfoEntity> MoneyChangerTracInfo { get; set; }
        public DbSet<MoneyTransferBankInfoEntity> MoneyTransferBankInfo { get; set; }
        public DbSet<MoneyTransferLogEntity> MoneyTransferLog { get; set; }
        public DbSet<MoneyTransferLogDetailEntity> MoneyTransferLogDetail { get; set; }
        public DbSet<MonthlyTransferAccountsEntity> MonthlyTransferAccounts { get; set; }
        public DbSet<MonthlyTransferConfigEntity> MonthlyTransferConfig { get; set; }
        public DbSet<MonthlyTransferGroupsEntity> MonthlyTransferGroups { get; set; }
        public DbSet<MoveHistoryDBJobConfigEntity> MoveHistoryDBJobConfig { get; set; }
        public DbSet<MpAgentsBetSettingEntity> MpAgentsBetSetting { get; set; }
        public DbSet<MpAgentsCommissionEntity> MpAgentsCommission { get; set; }
        public DbSet<MPAgentsTransferSettingEntity> MPAgentsTransferSetting { get; set; }
        public DbSet<MpCashSettledEntity> MpCashSettled { get; set; }
        public DbSet<MpCashUsedEntity> MpCashUsed { get; set; }
        public DbSet<MpLoginNameAvailabilityEntity> MpLoginNameAvailability { get; set; }
        public DbSet<MpPlayerBetSettingEntity> MpPlayerBetSetting { get; set; }
        public DbSet<MpPlayerCommissionEntity> MpPlayerCommission { get; set; }
        public DbSet<MpPTEffectiveEntity> MpPTEffective { get; set; }
        public DbSet<MpPTPresetEntity> MpPTPreset { get; set; }
        public DbSet<PartialCashOutRelationEntity> PartialCashOutRelation { get; set; }
        public DbSet<PatternLoginEntity> PatternLogin { get; set; }
        public DbSet<PhoneBetAccessEntity> PhoneBetAccess { get; set; }
        public DbSet<PhoneBettingAccountEntity> PhoneBettingAccount { get; set; }
        public DbSet<PlayerDiscountEntity> PlayerDiscount { get; set; }
        public DbSet<PlutoRepChecksumEntity> PlutoRepChecksum { get; set; }
        public DbSet<PlutoRepChecksumLogEntity> PlutoRepChecksumLog { get; set; }
        public DbSet<PresetPositionTakingEntity> PresetPositionTaking { get; set; }
        public DbSet<PTDirectAuditGroupMappingEntity> PTDirectAuditGroupMapping { get; set; }
        public DbSet<PTDirectEffectiveEntity> PTDirectEffective { get; set; }
        public DbSet<PTDirectPresetEntity> PTDirectPreset { get; set; }
        public DbSet<PTEffectiveEntity> PTEffective { get; set; }
        public DbSet<PTPresetEntity> PTPreset { get; set; }
        public DbSet<refEntity> _ref { get; set; }
        public DbSet<RejectBetsLogEntity> RejectBetsLog { get; set; }
        public DbSet<ResetProfileEntity> ResetProfile { get; set; }
        public DbSet<RiskControlBalanceEntity> RiskControlBalance { get; set; }
        public DbSet<RiskControlBalanceUpsertsEntity> RiskControlBalanceUpserts { get; set; }
        public DbSet<RoboticAccountEntity> RoboticAccount { get; set; }
        public DbSet<SalaryTransactionsEntity> SalaryTransactions { get; set; }
        public DbSet<SB_RobotBetIPEntity> SB_RobotBetIP { get; set; }
        public DbSet<SB_RobotBetQuarantineLogEntity> SB_RobotBetQuarantineLog { get; set; }
        public DbSet<SB_RobotSchedulerLogEntity> SB_RobotSchedulerLog { get; set; }
        public DbSet<SelfExclusionEmailNotifyEntity> SelfExclusionEmailNotify { get; set; }
        public DbSet<SensitiveInfoPermissionRequestEntity> SensitiveInfoPermissionRequest { get; set; }
        public DbSet<SettleActionEntity> SettleAction { get; set; }
        public DbSet<SettledBetTransEntity> SettledBetTrans { get; set; }
        public DbSet<SettledBetTransHistoryEntity> SettledBetTransHistory { get; set; }
        public DbSet<SettledOrderExtraInfoEntity> SettledOrderExtraInfo { get; set; }
        public DbSet<SettledOrderExtraInfo14Entity> SettledOrderExtraInfo14 { get; set; }
        public DbSet<SettledOrdersEntity> SettledOrders { get; set; }
        public DbSet<SettledOrders14Entity> SettledOrders14 { get; set; }
        public DbSet<SettledOrdersIntegrityCheckEntity> SettledOrdersIntegrityCheck { get; set; }
        public DbSet<SettlementLockEntity> SettlementLock { get; set; }
        public DbSet<SettlementTimeLogEntity> SettlementTimeLog { get; set; }
        public DbSet<SettleTypeDisplayTypeRelationEntity> SettleTypeDisplayTypeRelation { get; set; }
        public DbSet<ShareholderAuditGroupMappingEntity> ShareholderAuditGroupMapping { get; set; }
        public DbSet<sma_bal_bbuEntity> sma_bal_bbu { get; set; }
        public DbSet<SpecialSalaryTransactionsEntity> SpecialSalaryTransactions { get; set; }
        public DbSet<SportExclusionEntity> SportExclusion { get; set; }
        public DbSet<SportsBetExtraInfoEntity> SportsBetExtraInfo { get; set; }
        public DbSet<SportsRiskControlSettingEntity> SportsRiskControlSetting { get; set; }
        public DbSet<SSiteCustomerInfoEntity> SSiteCustomerInfo { get; set; }
        public DbSet<StatementEntity> Statement { get; set; }
        public DbSet<SuperSmaTagEntity> SuperSmaTag { get; set; }
        public DbSet<SuspendedByCompanyLogEntity> SuspendedByCompanyLog { get; set; }
        public DbSet<TableTrackerEntity> TableTracker { get; set; }
        public DbSet<TableTrackerBetSettingEntity> TableTrackerBetSetting { get; set; }
        public DbSet<TimeZonesEntity> TimeZones { get; set; }
        public DbSet<TracDelayEntity> TracDelay { get; set; }
        public DbSet<transEntity> trans { get; set; }
        public DbSet<TransferAgentEntity> TransferAgent { get; set; }
        public DbSet<TransferRequestEntity> TransferRequest { get; set; }
        public DbSet<UMCheckSumEntity> UMCheckSum { get; set; }
        public DbSet<UnsettledOrdersEntity> UnsettledOrders { get; set; }
        public DbSet<UpdatedForcePTDirectCustomerEntity> UpdatedForcePTDirectCustomer { get; set; }
        public DbSet<UpdatedPTEffectiveEntity> UpdatedPTEffective { get; set; }
        public DbSet<userlogEntity> userlog { get; set; }
        public DbSet<WongLaiAccountEntity> WongLaiAccount { get; set; }
        public DbSet<WonglaiSpecialAccountEntity> WonglaiSpecialAccount { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.ApplyConfiguration(new __RefactorLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AccountActionLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AccountingCheckSumEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AccountingDateEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AccountSettleBetTempEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AccountSuspendLimitEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AgentFeatureEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AgentMappingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AgentPatternLoginEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AgentsBetSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AgentsDiscountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AgentsMaxPTEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AgentsMonitorListEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AgentsTransferSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new APIAccountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AuditCheckedListEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AuditGroupMoveLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AutoCBSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AutoSettleLeagueListEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AutoSettlementEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AutoSettlementRecordEntityConfiguration());
            modelBuilder.ApplyConfiguration(new AutoSuspendLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new B2B2CRebateReportEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BankAccountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BankGroupBankInfoEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BankGroupMoneyChangerSenderInfoEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BankGroupPartnershipSenderInfoEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BatchLockEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BatchResultHistoryEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BetBuilderLegEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BetBuilderLeg14EntityConfiguration());
            modelBuilder.ApplyConfiguration(new bettransEntityConfiguration());
            modelBuilder.ApplyConfiguration(new bettrans14EntityConfiguration());
            modelBuilder.ApplyConfiguration(new bettransmEntityConfiguration());
            modelBuilder.ApplyConfiguration(new bettransm14EntityConfiguration());
            modelBuilder.ApplyConfiguration(new BlackListWordEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BlindRiskCustomerEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BlindRiskCustomerClusterEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BlindRiskCustomerDailyBetSummaryEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BonusTransactionsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BonusWalletEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BonusWalletCashSettledEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BonusWalletCashUsedEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BonusWalletDailyStatementEntityConfiguration());
            modelBuilder.ApplyConfiguration(new BonusWalletStatementEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CancelledMatchEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashOutActionLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashOutBettransEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashOutBettransmEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashSettledEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashSettledLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashSettledLogSumEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashSettledTransactionEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashSettledUpsertEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashUsedEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashUsedLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CashUsedLogSumEntityConfiguration());
            modelBuilder.ApplyConfiguration(new ChangeParentLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CitiesEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CompanyRiskEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CorporateCustomerLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CountriesEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CountryPhoneCodeEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CoverBetAccountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CurrencyTransactionsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new custcontrolEntityConfiguration());
            modelBuilder.ApplyConfiguration(new customerEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerCreditEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerGroupEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerGroupCreditSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerMappingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerPromotionEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerPromotionConfigEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerPromotionSignUpEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerRiskLimitEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerSettingVerificationLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerTournamentEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerTrackingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new CustomerVvipGroupEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DailyCBEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DailyStatementEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DailyStatementSumEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DailyStatementTransactionEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DailyStatementUpsertsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DBActionLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DBErrorLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DeletedCustomerEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DeletedStmtLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DirectCustomerAuditEntityConfiguration());
            modelBuilder.ApplyConfiguration(new DisplayNamePrefixEntityConfiguration());
            modelBuilder.ApplyConfiguration(new EarlySettlementEntityConfiguration());
            modelBuilder.ApplyConfiguration(new EffectivePostionTakingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new EnumEntityConfiguration());
            modelBuilder.ApplyConfiguration(new exchangeEntityConfiguration());
            modelBuilder.ApplyConfiguration(new ExchangeCodeLookUpEntityConfiguration());
            modelBuilder.ApplyConfiguration(new ExternalExchangeEntityConfiguration());
            modelBuilder.ApplyConfiguration(new FailedSettlementBetsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new FantasyDailyWinLostEntityConfiguration());
            modelBuilder.ApplyConfiguration(new FantasyRankingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new FantasySportsBetEntityConfiguration());
            modelBuilder.ApplyConfiguration(new FantasySportsOrderEntityConfiguration());
            modelBuilder.ApplyConfiguration(new FollowBetAccountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new GamePreOddsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new GamesCashUsedTransactionEntityConfiguration());
            modelBuilder.ApplyConfiguration(new IomCustomersAllowedForSBOProductsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new JoinNowPromotionEntityConfiguration());
            modelBuilder.ApplyConfiguration(new JoinNowPromotionLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new KYCEmailNotifyEntityConfiguration());
            modelBuilder.ApplyConfiguration(new LockAccountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new LoginBlackListEntityConfiguration());
            modelBuilder.ApplyConfiguration(new LoginWhiteListEntityConfiguration());
            modelBuilder.ApplyConfiguration(new LuckyBetEntityConfiguration());
            modelBuilder.ApplyConfiguration(new matchEntityConfiguration());
            modelBuilder.ApplyConfiguration(new match_buffertableEntityConfiguration());
            modelBuilder.ApplyConfiguration(new match14EntityConfiguration());
            modelBuilder.ApplyConfiguration(new MatchChangeLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MatchResultEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MatchResult_14_BufferTableEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MatchResult_BufferTableEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MatchResult14EntityConfiguration());
            modelBuilder.ApplyConfiguration(new MatchResult14_BufferTableEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MaxBetReducedAgentsPlayersIdEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MaxPTConfigEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MemberBetSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MemberProdSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MnlStatementForIomCustomerEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MnlStatementForIomCustomerRefEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MoneyChangerTracInfoEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MoneyTransferBankInfoEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MoneyTransferLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MoneyTransferLogDetailEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MonthlyTransferAccountsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MonthlyTransferConfigEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MonthlyTransferGroupsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MoveHistoryDBJobConfigEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MpAgentsBetSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MpAgentsCommissionEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MPAgentsTransferSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MpCashSettledEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MpCashUsedEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MpLoginNameAvailabilityEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MpPlayerBetSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MpPlayerCommissionEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MpPTEffectiveEntityConfiguration());
            modelBuilder.ApplyConfiguration(new MpPTPresetEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PartialCashOutRelationEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PatternLoginEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PhoneBetAccessEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PhoneBettingAccountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PlayerDiscountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PlutoRepChecksumEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PlutoRepChecksumLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PresetPositionTakingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PTDirectAuditGroupMappingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PTDirectEffectiveEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PTDirectPresetEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PTEffectiveEntityConfiguration());
            modelBuilder.ApplyConfiguration(new PTPresetEntityConfiguration());
            modelBuilder.ApplyConfiguration(new refEntityConfiguration());
            modelBuilder.ApplyConfiguration(new RejectBetsLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new ResetProfileEntityConfiguration());
            modelBuilder.ApplyConfiguration(new RiskControlBalanceEntityConfiguration());
            modelBuilder.ApplyConfiguration(new RiskControlBalanceUpsertsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new RoboticAccountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SalaryTransactionsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SB_RobotBetIPEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SB_RobotBetQuarantineLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SB_RobotSchedulerLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SelfExclusionEmailNotifyEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SensitiveInfoPermissionRequestEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettleActionEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettledBetTransEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettledBetTransHistoryEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettledOrderExtraInfoEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettledOrderExtraInfo14EntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettledOrdersEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettledOrders14EntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettledOrdersIntegrityCheckEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettlementLockEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettlementTimeLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SettleTypeDisplayTypeRelationEntityConfiguration());
            modelBuilder.ApplyConfiguration(new ShareholderAuditGroupMappingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new sma_bal_bbuEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SpecialSalaryTransactionsEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SportExclusionEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SportsBetExtraInfoEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SportsRiskControlSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SSiteCustomerInfoEntityConfiguration());
            modelBuilder.ApplyConfiguration(new StatementEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SuperSmaTagEntityConfiguration());
            modelBuilder.ApplyConfiguration(new SuspendedByCompanyLogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new TableTrackerEntityConfiguration());
            modelBuilder.ApplyConfiguration(new TableTrackerBetSettingEntityConfiguration());
            modelBuilder.ApplyConfiguration(new TimeZonesEntityConfiguration());
            modelBuilder.ApplyConfiguration(new TracDelayEntityConfiguration());
            modelBuilder.ApplyConfiguration(new transEntityConfiguration());
            modelBuilder.ApplyConfiguration(new TransferAgentEntityConfiguration());
            modelBuilder.ApplyConfiguration(new TransferRequestEntityConfiguration());
            modelBuilder.ApplyConfiguration(new UMCheckSumEntityConfiguration());
            modelBuilder.ApplyConfiguration(new UnsettledOrdersEntityConfiguration());
            modelBuilder.ApplyConfiguration(new UpdatedForcePTDirectCustomerEntityConfiguration());
            modelBuilder.ApplyConfiguration(new UpdatedPTEffectiveEntityConfiguration());
            modelBuilder.ApplyConfiguration(new userlogEntityConfiguration());
            modelBuilder.ApplyConfiguration(new WongLaiAccountEntityConfiguration());
            modelBuilder.ApplyConfiguration(new WonglaiSpecialAccountEntityConfiguration());
        }
    }
}
