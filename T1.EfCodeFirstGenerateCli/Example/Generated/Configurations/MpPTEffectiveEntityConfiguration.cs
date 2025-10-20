using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MpPTEffectiveEntityConfiguration : IEntityTypeConfiguration<MpPTEffectiveEntity>
    {
        public void Configure(EntityTypeBuilder<MpPTEffectiveEntity> builder)
        {
            builder.ToTable("MpPTEffective");

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

            builder.Property(x => x.RoleId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MinimumPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ForcedPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.TakeRemaining)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.PT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
