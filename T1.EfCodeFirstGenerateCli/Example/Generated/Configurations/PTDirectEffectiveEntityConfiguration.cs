using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class PTDirectEffectiveEntityConfiguration : IEntityTypeConfiguration<PTDirectEffectiveEntity>
    {
        public void Configure(EntityTypeBuilder<PTDirectEffectiveEntity> builder)
        {
            builder.ToTable("PTDirectEffective");

            builder.HasKey(x => new { x.CustomerId, x.ParentId });

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentMinimum)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ParentForce)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ParentEffective)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedBy)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LastModifiedDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
