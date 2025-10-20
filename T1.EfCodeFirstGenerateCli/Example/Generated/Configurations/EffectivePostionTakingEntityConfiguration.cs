using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class EffectivePostionTakingEntityConfiguration : IEntityTypeConfiguration<EffectivePostionTakingEntity>
    {
        public void Configure(EntityTypeBuilder<EffectivePostionTakingEntity> builder)
        {
            builder.ToTable("EffectivePostionTaking");

            builder.HasKey(x => new { x.PlayerId, x.AgentId, x.Type });

            builder.Property(x => x.PlayerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.AgentId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
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

            builder.Property(x => x.Effective)
                .HasColumnType("decimal(3,2)")
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
