using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class PresetPositionTakingEntityConfiguration : IEntityTypeConfiguration<PresetPositionTakingEntity>
    {
        public void Configure(EntityTypeBuilder<PresetPositionTakingEntity> builder)
        {
            builder.ToTable("PresetPositionTaking");

            builder.HasKey(x => new { x.CustomerId, x.Type });

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Type)
                .HasColumnType("tinyint(3,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ParentForce)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.TakeRemaining)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
