using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerPromotionSignUpEntityConfiguration : IEntityTypeConfiguration<CustomerPromotionSignUpEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerPromotionSignUpEntity> builder)
        {
            builder.ToTable("CustomerPromotionSignUp");


            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.PromotionType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Option)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CurrentWinlost)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.HistoryWinlost)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.CurrentBetCount)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.HistoryBetCount)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.JoinDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.PromotionCategory)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.Currency)
                .HasColumnType("varchar(3)")
                .HasMaxLength(3)
            ;

            builder.Property(x => x.LoginName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CurrentTotalBetCount)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.HistoryTotalBetCount)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

        }
    }
}
