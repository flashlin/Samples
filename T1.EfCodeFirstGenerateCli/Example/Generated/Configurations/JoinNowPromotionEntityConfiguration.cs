using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class JoinNowPromotionEntityConfiguration : IEntityTypeConfiguration<JoinNowPromotionEntity>
    {
        public void Configure(EntityTypeBuilder<JoinNowPromotionEntity> builder)
        {
            builder.ToTable("JoinNowPromotion");

            builder.HasKey(x => x.ID);

            builder.Property(x => x.ID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.UserName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ISOCurrency)
                .HasColumnType("char(3)")
                .IsRequired()
                .HasMaxLength(3)
                .HasDefaultValue("")
            ;

            builder.Property(x => x.PromotionType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TargetTurnOver)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.TurnOver14)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.WinLoss14)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.BonusAmount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.ExpiryDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.PromotionStatus)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(255)")
                .HasMaxLength(255)
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.PromotionCode)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.MaxEntitlement)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.EntitlementRate)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.LiveIndicator)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.supportedmarkettype)
                .HasColumnType("varchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.AdminFeeRate)
                .HasColumnType("decimal(5,2)")
            ;

            builder.Property(x => x.isHitTarget)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.FirstBetTransId)
                .HasColumnType("bigint(19,0)")
            ;

            builder.Property(x => x.CreditedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
