using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BonusWalletEntityConfiguration : IEntityTypeConfiguration<BonusWalletEntity>
    {
        public void Configure(EntityTypeBuilder<BonusWalletEntity> builder)
        {
            builder.ToTable("BonusWallet");

            builder.HasKey(x => x.BonusWalletId);

            builder.Property(x => x.BonusWalletId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.GroupKey)
                .HasColumnType("varchar(250)")
                .IsRequired()
                .HasMaxLength(250)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
