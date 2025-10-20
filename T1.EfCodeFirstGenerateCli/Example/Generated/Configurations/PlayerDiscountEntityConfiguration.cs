using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class PlayerDiscountEntityConfiguration : IEntityTypeConfiguration<PlayerDiscountEntity>
    {
        public void Configure(EntityTypeBuilder<PlayerDiscountEntity> builder)
        {
            builder.ToTable("PlayerDiscount");

            builder.HasKey(x => x.CustID);

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.AgtID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MaID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.SmaID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.PlayerDiscount)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.AgtDiscount)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaDiscount)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaDiscount)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.PlayerDiscount1x2)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.AgtDiscount1x2)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaDiscount1x2)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaDiscount1x2)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.PlayerDiscountOther)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.AgtDiscountOther)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaDiscountOther)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaDiscountOther)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.Ugroup)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

        }
    }
}
