using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AgentsDiscountEntityConfiguration : IEntityTypeConfiguration<AgentsDiscountEntity>
    {
        public void Configure(EntityTypeBuilder<AgentsDiscountEntity> builder)
        {
            builder.ToTable("AgentsDiscount");

            builder.HasKey(x => x.CustID);

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.DiscountOther)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.DiscountGroupA)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.DiscountGroupB)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.DiscountGroupC)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.Discount1x2)
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

        }
    }
}
