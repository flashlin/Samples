using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AccountSuspendLimitEntityConfiguration : IEntityTypeConfiguration<AccountSuspendLimitEntity>
    {
        public void Configure(EntityTypeBuilder<AccountSuspendLimitEntity> builder)
        {
            builder.ToTable("AccountSuspendLimit");

            builder.HasKey(x => x.CustomerId);

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.SuspendLimit)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.IsEnabled)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.AccountId)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
