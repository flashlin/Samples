using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class FollowBetAccountEntityConfiguration : IEntityTypeConfiguration<FollowBetAccountEntity>
    {
        public void Configure(EntityTypeBuilder<FollowBetAccountEntity> builder)
        {
            builder.ToTable("FollowBetAccount");

            builder.HasKey(x => x.CustID);

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.AccountID)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ISOCurrency)
                .HasColumnType("varchar(3)")
                .HasMaxLength(3)
            ;

            builder.Property(x => x.FollowCustID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.FollowAccountID)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.IsEnabled)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.FollowPercentage)
                .HasColumnType("decimal(12,2)")
                .IsRequired()
                .HasDefaultValue(1)
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
