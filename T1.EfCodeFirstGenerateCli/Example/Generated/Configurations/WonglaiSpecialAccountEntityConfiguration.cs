using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class WonglaiSpecialAccountEntityConfiguration : IEntityTypeConfiguration<WonglaiSpecialAccountEntity>
    {
        public void Configure(EntityTypeBuilder<WonglaiSpecialAccountEntity> builder)
        {
            builder.ToTable("WonglaiSpecialAccount");

            builder.HasKey(x => x.CustomerId);

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.AccountId)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LastLoginDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.TransactionCutOffDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastLoginIp)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LastLoginCountry)
                .HasColumnType("char(2)")
                .HasMaxLength(2)
            ;

            builder.Property(x => x.LastTransactionDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.OldExtraInfoId)
                .HasColumnType("int(10,0)")
            ;

        }
    }
}
